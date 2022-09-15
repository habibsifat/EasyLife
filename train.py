import atexit
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch import tensor

from device import device
from game import generate_random_board
from game import life_step


def train(model,
          batch_size=100,
          grid_size=25,
          accuracy_count=100_000,
          l1=0,
          l2=0,
          timeout=0,
          reverse_input_output=False,
          epochs=0,
          verbose=True,
):
    assert 1 <= grid_size <= 25

    if verbose: print(f'{model.device} Training: {model.__class__.__name__}')
    time_start = time.perf_counter()

    atexit.register(model.save)      
    model.load(verbose=verbose).train().unfreeze()  
    if verbose: print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = None

    

    num_params = torch.sum(torch.tensor([
        torch.prod(torch.tensor(param.shape))
        for param in model.parameters()
    ]))

    epoch        = 0
    board_count  = 0
    last_loss    = np.inf
    loop_loss    = 0
    loop_acc     = 0
    loop_count   = 0
    epoch_losses     = [last_loss]
    epoch_accuracies = [ 0 ]

    
    try:
        epoch_time = 0
        for epoch in range(1, sys.maxsize):
            if epochs and epochs < epoch:                                      break
            if np.min(epoch_accuracies[-accuracy_count//batch_size:]) == 1.0:  break  # multiple epochs of 100% accuracy to pass
            if timeout and timeout < time.perf_counter() - time_start:         break
            epoch_start = time.perf_counter()

            inputs_np   = [ generate_random_board(shape=(grid_size,grid_size)) for _     in range(batch_size) ]
            expected_np = [ life_step(board)                                   for board in inputs_np ]
            inputs      = model.cast_inputs(inputs_np).to(device)
            expected    = model.cast_inputs(expected_np).to(device)

            if reverse_input_output:
                inputs, expected = expected, inputs

            
            for _ in range(5):  # repeat each dataset 5 times
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = model.loss(outputs, expected, inputs)
                if l1 or l2:
                    l1_loss = torch.sum(tensor([ torch.sum(torch.abs(param)) for param in model.parameters() ])) / num_params
                    l2_loss = torch.sum(tensor([ torch.sum(param**2)         for param in model.parameters() ])) / num_params
                    loss   += ( l1_loss * l1 ) + ( l2_loss * l2 )

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    # scheduler.step(loss)  # only required for
                    scheduler.step()

                
                last_accuracy = model.accuracy(outputs, expected, inputs)  
                last_loss     = loss.item() # / batch_size

                epoch_losses.append(last_loss)
                epoch_accuracies.append( last_accuracy )

                loop_loss   += last_loss
                loop_acc    += last_accuracy
                loop_count  += 1
                board_count += batch_size
                epoch_time   = time.perf_counter() - epoch_start

            if (epoch <= 10) or (epoch <= 100 and epoch % 10 == 0) or epoch % 100 == 0:
                time_taken = time.perf_counter() - time_start
                if verbose: print(f'{epoch:4d} | {board_count:7d} | loss: {loop_loss/loop_count:.10f} | accuracy = {loop_acc/loop_count:.10f} | {1000*epoch_time/batch_size:6.3f}ms/board | {time_taken//60:3.0f}m {time_taken%60:02.0f}s ')
                loop_loss  = 0
                loop_acc   = 0
                loop_count = 0

    except (BrokenPipeError, KeyboardInterrupt):
        pass
    except Exception as exception:
        print(exception)
        raise exception
    finally:
        time_taken = time.perf_counter() - time_start
        if verbose: print(f'Finished Training: {model.__class__.__name__} - {epoch} epochs in {time_taken:.1f}s')
        model.save(verbose=verbose)
        atexit.unregister(model.save)   # model now saved, so cancel atexit handler
        # model.eval()                  # disable dropout

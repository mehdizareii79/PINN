'''PINN training.'''

from warnings import warn


def test_pinn(pinn, colloc_dict):
    '''
    Test PINN physics loss.

    Summary
    -------
    The physics loss of a PINN is computed for given collocation points.
    It is remarked that, due to the occurrence of the partial derivatives
    in the loss function, the autograd machinery needs to be enabled.

    Parameters
    ----------
    pinn : PINN module
        PINN model with a physics loss method.
    colloc_dict : dict
        Dict of collocation points.

    '''
    #put model in eval mode
    ##############################
    #TODO#
    ##############################
    #calculate its loss
    ##############################
    #TODO#
    ##############################
    return loss


def train_pinn(pinn,
               optimizer,
               num_epochs,
               train_colloc,
               val_colloc=None,
               print_every=1):
    '''
    Train PINN by minimizing the physics loss.

    Summary
    -------
    A CPU-based non-batched training scheme for PINNs is provided.
    The physics loss is minimized for a given set of collocation points.
    It is assumed that no real observational data is available,
    such that the regression loss can be omitted.

    Parameters
    ----------
    pinn : PINN module
        PINN model with a physics loss method.
    num_epochs : int
        Number of training epochs.
    train_colloc : dict
        Dict of collocation points for training.
    val_colloc : dict
        Dict of collocation points for validation.
    print_every : int
        Determines when losses are printed.

    '''
    history = {'Train loss': [], 'Val loss': []}
    pinn.train()
    for epoch in range(num_epochs):
        loss = pinn.physics_loss(train_colloc['pde_data'], train_colloc['bc_data'], train_colloc['ic_data'])
        val_loss = pinn.physics_loss(val_colloc['pde_data'], val_colloc['bc_data'], val_colloc['ic_data'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history['Train loss'].append(loss.item())
        history['Val loss'].append(val_loss.item())
        if epoch % print_every == 0:
           print(f' Epoch {epoch},Training Loss : {loss.item()} , Validation Loss: {val_loss}')
    return history


class Sgd():

    def __init__(self,learning_rate:float) -> None:
        self.learning_rate = learning_rate
        pass

    #calculate update(weight tensor, gradient tensor) that returns the updated weights
    #according to the basic gradient descent update scheme.
    def calculate_update(self,weight_tensor, gradient_tensor):
        new_weight = weight_tensor - gradient_tensor * self.learning_rate
        return new_weight

    
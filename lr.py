
from numpy import *

def compute_error_for_line_given_points(b, m, points):
    # Mean Squared Error (https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x +b)) **2
    return totalError / float(len(points))

def step_gradient(b_current, m_current , points, learningRate):
    #Gradient Descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range (0, len(points)):
        # Calculate partial derivatives for b an m (https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
        x = points[i, 0]
        y = points[i, 1]
        # -2(y - (mx + b))
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        # -2x(y - (mx + b))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_epoch):
    b = starting_b
    m = starting_m

    for i in range(num_epoch):
        b, m = step_gradient (b, m, array(points), learning_rate)
    return [b, m]

def run():
    '''
    The point of this is to demonstrate the concept of gradient descent.
    Gradient descent is the most popular optimization strategy in deep learning, in particular an implementation of it called backpropagation.
    We are using gradient descent as our optimization strategy for linear regression.
    '''

    points = genfromtxt('data.csv', delimiter=',')
    #hypermater
    learning_rate = 0.0001
    # from algebra slope formula is y = mx + b
    initial_b = 0
    initial_m = 0
    num_epoch = 100

    print ('Starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ('Running...')
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_epoch)
    print ('After {0} iterations b = {1}, m = {2}, error = {3}'.format(num_epoch, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()

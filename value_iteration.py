import numpy as np


def one_step_lookahead(environment, state, V, discount_factor):
    action_values = np.zeros(environment.nA)
    for action in range(environment.nA):

         #stochasticity
         for probability, next_state, reward, finished in environment.P[state][action]:
             action_values[action] += probability * (reward + discount_factor * V[next_state])

    return action_values

def value_iteration(environment, discount_factor=1.0, epsilon=1e-9, max_iterations=1e6):
    V = np.random.rand(environment.nS)

    cpt_iterations = 0
    delta = epsilon + 1

    while cpt_iterations < max_iterations and delta >= epsilon:
        delta = 0

        for state in range(environment.nS):
            action_value = one_step_lookahead(environment, state, V, discount_factor)
            best_action_value = np.max(action_value)
            # probabilities = action_value + 1
            # probabilities = np.exp(probabilities)
            # probabilities /= probabilities.sum()
            # best_action_value = np.random.choice(action_value, p=probabilities)
            delta = max(delta, np.abs(V[state] - best_action_value))
            V[state] = best_action_value

        cpt_iterations += 1
    
    if delta < epsilon:
        print('Converged at iteraction n°{}.'.format(cpt_iterations))
    elif cpt_iterations == max_iterations:
        print('Max number of iterations reached, no convergence...')
    
    policy = np.zeros(environment.nS)
    for state in range(environment.nS):
        action_value = one_step_lookahead(environment, state, V, discount_factor)
        policy[state] = np.argmax(action_value)

    return V, policy


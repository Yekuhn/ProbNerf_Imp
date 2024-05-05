def temperature_annealing(iteration, total_iterations):
    t_initial = 1.0
    t_final = 0.1
    return t_initial * (t_final / t_initial) ** (iteration / total_iterations)

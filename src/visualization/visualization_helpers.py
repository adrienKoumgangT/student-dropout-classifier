import os

figures_path = os.path.join(os.getcwd(), 'reports', 'figures')

def form_path(filename: str):
    return os.path.join(figures_path, filename)



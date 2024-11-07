class ResultsDirException(Exception):
    """
    Raised when the results directory already exists
    """
    def __init__(self, exp, results_dir='./logs'):
        message = f"\n" \
                  f"When trying to add the experiment: {exp}\n" \
                  f"Results directory {results_dir} already exists.\n" \
                  f"Make sure that varying parameters have a trailing double underscore."
        super().__init__(message)

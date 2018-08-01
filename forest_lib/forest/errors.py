class NoDataError(IOError):
    """
    Exception for unable to access the data for any reason.
    """

    def __init__(self, original_error=None):
        """
        Create an NoDataError

        Args:
            original_error (Exception) : Original error if there is one.
        """
        error = getattr(original_error, "args", (None, None))
        error = (error[0], error[1], original_error.filename) if original_error.filename else error
        super().__init__(error)

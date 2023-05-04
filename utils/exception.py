
class NotImplementException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info
        super().__init__(self)

    def __str__(self):
        return self.error_info

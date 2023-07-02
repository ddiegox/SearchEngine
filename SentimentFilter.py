from whoosh.query import NumericRange


class SentimentFilter(NumericRange):
    sentiment_filter = None

    def __init__(self, mode):
        if mode >= 0:  # positive review filter
            super().__init__("sentiment", 0, None, startexcl=True)
        elif mode <= -0:  # negative review filter
            super().__init__("sentiment", None, 0, endexcl=True)
        else:  # no filter
            super().__init__("sentiment", None, None)
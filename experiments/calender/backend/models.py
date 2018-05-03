from datetime import datetime

class Event(object):
    def __init__(self, name: str, begin: datetime, end: datetime, summary: str):
        self.name = name
        self.begin = begin
        self.end = end
        self.summary = summary

    
    
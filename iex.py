import requests

class IEXStock:

    def __init__(self, token, symbol, environment='production'):
        if environment == 'production':
            self.BASE_URL = 'https://cloud.iexapis.com/v1'
        else:
            self.BASE_URL = 'https://sandbox.iexapis.com/v1'
        
        self.token = token
        self.symbol = symbol

    def get_logo(self):
        url = f"{self.BASE_URL}/stock/{self.symbol}/logo?token={self.token}"
        r = requests.get(url)

        return r.json()

    def get_company_info(self):
        url = f"{self.BASE_URL}/stock/{self.symbol}/company?token={self.token}"
        r = requests.get(url)

        return r.json()    
    def get_company_news(self, last=10):
        url = f"{self.BASE_URL}/stock/{self.symbol}/news/last/{last}?token={self.token}"
        r = requests.get(url)

        return r.json()

#        def get_insider_transactions(self):
#            url = f"{self.BASE_URL}/stock/{self.symbol}/insider-transactions?token={self.token}"
#            r = requests.get(url)
#
#            return r.json()



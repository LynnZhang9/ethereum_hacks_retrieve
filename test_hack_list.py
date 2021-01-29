import requests

# YourApiKeyToken = { "key" : "258XQKQW5UWY7MYXR9TX7FC5A6ZZXU1PBP" }
# address = "0xfE6858d6C72dc21Ea31463F23c973A6ab56cdCB6"
# url = "http://api.etherscan.io/api?module=account&action=txlist&address="+ address + \
#       "&startblock=0&endblock=99999999&page=1&offset=10&sort=asc&apikey=YourApiKeyToken"
# url = "https://etherscan.io/accounts/label/phish-hack"
# url = "https://api.cryptoscamdb.org/v1/scams"
url = "https://api.cryptoscamdb.org/v1/addresses"
# url = "https://api.cryptoscamdb.org/v1/blacklist"
response = requests.get(url)
result = response.json().get("result")
count = 0
for i, address in enumerate(result):
    # The attribut you want

    type = result[address][0]["type"]
    address = result[address][0]["address"]
    category = result[address][0]["category"]
    # if "Phishing" == category:
    if "scam" == type:
        print("Transanction ID {}".format(i))
        print("type: {}".format(type))
        print("address: {}".format(address))
        print("category: {}".format(category))
        print(count)
        print("\n")
        count += 1

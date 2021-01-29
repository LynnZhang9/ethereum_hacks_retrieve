import requests

YourApiKeyToken = { "key" : "258XQKQW5UWY7MYXR9TX7FC5A6ZZXU1PBP" }
address = "0xfE6858d6C72dc21Ea31463F23c973A6ab56cdCB6"
url = "http://api.etherscan.io/api?module=account&action=txlist&address="+ address + \
      "&startblock=0&endblock=99999999&page=1&offset=10&sort=asc&apikey=YourApiKeyToken"


response = requests.get(url)
result = response.json().get("result")

for i, transaction in enumerate(result):
    # The attribut you want
    block_num = transaction.get("blockNumber")
    timestamp = transaction.get("timeStamp")
    hash = transaction.get("hash")
    tx_from = transaction.get("from")
    tx_to = transaction.get("to")
    value = transaction.get("value")
    print("Transanction ID {}. block_num: {}".format(i, block_num))
    print("Transanction ID {}. timestamp: {}".format(i, timestamp))
    print("Transanction ID {}. hash: {}".format(i, hash))
    print("Transanction ID {}. tx_from: {}".format(i, tx_from))
    print("Transanction ID {}. tx_to: {}".format(i, tx_to))


    print("Transanction ID {}. value: {}".format(i, value))
    print("\n")

import urllib.request
import json


def translate(caption):
    client_id = "***************" # 개발자센터에서 발급받은 Client ID 값
    client_secret = "********" # 개발자센터에서 발급받은 Client Secret 값
    kocText = urllib.parse.quote(caption)
    data = "source=en&target=ko&text=" + kocText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        result = json.loads(response_body.decode('utf-8'))
        result = result['message']['result']['translatedText']
    else:
        print("Error Code:" + rescode)
    
    return result




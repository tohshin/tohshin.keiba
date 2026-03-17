import chardet

f_path = r"c:\Users\kyoui\tohshin_keiba\deploy_tmp\index.html"
with open(f_path, 'rb') as f:
    rawdata = f.read()
    result = chardet.detect(rawdata)
    print(result)

import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
import time
import datetime


def connect_gspread(jsonf, key):
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    #認証情報の設定
    credentials = ServiceAccountCredentials.from_json_keyfile_name(jsonf, scope)
    gc = gspread.authorize(credentials)
    #スプレッドシートキーを用いてsheet1にアクセス
    SPREADSHEET_KEY = key
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    return worksheet

#jsonfile名を指定
jsonf = 'neural-gantry-317013-6433c562b3a7.json'
#共有設定したスプレッドシートキーを指定
spread_sheet_key = '1CDUOHL39xevPz1gUSvompFFcAgtf8ts9A9cb-Oo1VVQ'
ws = connect_gspread(jsonf, spread_sheet_key)

#Spread Sheets上の値を取得
values = ws.get_all_values()
row_num = len(values)
col_num = len(values[0])
print(row_num, col_num)

#Spread Sheets上の値を更新
date_check = values[row_num-1][0]
days = datetime.date.today()
now = datetime.datetime.now()
d_time = str(now.hour) + ':' + str(now.minute)

#日付の書き込み
if(date_check != str(days)):
    ws.update_cell(row_num+1,1,str(days))
    ws.update_cell(row_num+1,2,str(d_time))
else:
    ws.update_cell(row_num,1,str(days))
    ws.update_cell(row_num,3,str(d_time))


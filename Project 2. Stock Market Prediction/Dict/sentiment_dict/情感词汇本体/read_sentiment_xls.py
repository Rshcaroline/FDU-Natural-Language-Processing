#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/9 下午7:52
# @Author  : Shihan Ran
# @Site    : 
# @File    : read_sentiment_xls.py
# @Software: PyCharm

# this .py is aimed to read "情感词汇.xlsx"

# -*- coding: utf-8 -*-
import  xdrlib ,sys
import xlrd

def open_excel(file= './情感词汇.xlsx'):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception:
        print('error')

# 根据索引获取Excel表格中的数据
# 参数:file：Excel文件路径
# colnameindex：表头列名所在行
# by_index：表的索引
def excel_table_byindex(file= './情感词汇.xlsx', colnameindex=0, by_index=0):
    data = open_excel(file)
    table = data.sheets()[by_index]
    nrows = table.nrows #行数
    ncols = table.ncols #列数
    colnames =  table.row_values(colnameindex) #某一行数据
    list =[]
    for rownum in range(1,nrows):

         row = table.row_values(rownum)
         if row:
             app = {}
             for i in range(len(colnames)):
                app[colnames[i]] = row[i]
             list.append(app)
    return list

# 根据名称获取Excel表格中的数据
# 参数:file：Excel文件路径
# colnameindex：表头列名所在行的所以
# by_name：Sheet1名称
def excel_table_byname(file= './情感词汇.xlsx',colnameindex=0, by_name=u'Sheet1'):
    data = open_excel(file)
    table = data.sheet_by_name(by_name)
    nrows = table.nrows #行数
    colnames =  table.row_values(colnameindex) #某一行数据
    list =[]
    for rownum in range(1,nrows):
         row = table.row_values(rownum)
         if row:
             app = {}
             for i in range(len(colnames)):
                app[colnames[i]] = row[i]
             list.append(app)
    return list

def main():
   tables = excel_table_byindex()
   for row in tables:
       print(row)

   tables = excel_table_byname()
   for row in tables:
       print(row)

if __name__=="__main__":
    # main()

    pospath = './pos_word.txt'
    negpath = './neg_word.txt'
    posfile = open(pospath, 'w')
    negfile = open(negpath, 'w')

    tables = excel_table_byindex()
    for row in tables:
        if int(row['极性']) == 2: negfile.write(str(row['词语']) + ',' + str(row['强度']) + '\n')
        else: posfile.write(str(row['词语']) + ',' + str(row['强度']) + '\n')

    posfile.close()
    negfile.close()
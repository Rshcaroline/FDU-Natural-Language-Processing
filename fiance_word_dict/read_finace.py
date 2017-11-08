#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/8 下午7:52
# @Author  : Shihan Ran
# @Site    : 
# @File    : read_finace.py
# @Software: PyCharm

# this .py is aimed to read "金融词汇（英汉词典）.xls"

# -*- coding: utf-8 -*-
import  xdrlib ,sys
import xlrd

def open_excel(file= './金融词汇(英汉词典).xls'):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception:
        print('error')

#根据索引获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_index：表的索引
def excel_table_byindex(file= './金融词汇(英汉词典).xls', colnameindex=0,by_index=0):
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

#根据名称获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_name：Sheet1名称
def excel_table_byname(file= './金融词汇(英汉词典).xls',colnameindex=0, by_name=u'Sheet1'):
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

    resultpath = './finace word xls.txt'
    resultfile = open(resultpath, 'w')

    tables = excel_table_byindex()
    for row in tables:
        if row['中文翻译']:
            # print()
            resultfile.write(row['中文翻译'] + '\n')

    resultfile.close()
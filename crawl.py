#-*-coding:utf-8-*-
#file: crawl.py
#@author: hwh
#@contact: ruc_hwh_2013@163.com
#Created on 2018/1/19 下午11:23

'''
爬取网易云音乐流行和古风两种类型的歌曲
'''

from selenium import webdriver
from bs4 import BeautifulSoup
import xlwt
from multiprocessing import Pool

def Crawl(urls,label):
    results={}
    for url in urls:
        driver = webdriver.PhantomJS()
        driver.get(url)
        driver.switch_to_frame('g_iframe')
        page = driver.page_source
        soup = BeautifulSoup(page, 'html.parser')
        body = soup.find('tbody')
        lists = body.find_all('tr')
        for l in lists:
            span = l.find('span', class_='txt')
            title = span.find('b')['title']
            url = 'http://music.163.com/#' + span.find('a')['href']
            print title,url
            results.setdefault(title,url+'\t'+label)
    return results

def Crawl_lyrics(url):
    driver = webdriver.PhantomJS()
    driver.get(url)
    driver.switch_to_frame('g_iframe')
    page = driver.page_source
    soup = BeautifulSoup(page, 'html.parser')
    content=soup.find('div', id='lyric-content').get_text()
    print 'success'
    return content

def write_to_excel(results,path):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(u'网易云音乐歌词')
    worksheet.write(0, 0, 'name')
    worksheet.write(0, 1, 'lyrics')
    worksheet.write(0, 2, 'label')
    flag = 1
    for key, value in results.items():
        worksheet.write(flag, 0, key)
        url = value.strip().split('\t')[0]
        lyrics = Crawl_lyrics(url).split(':')[-1]
        label = value.strip().split('\t')[1]
        worksheet.write(flag, 1, lyrics)
        worksheet.write(flag, 2, label)
        flag += 1
    workbook.save(path)


if __name__=='__main__':
    liuxing_urls=['http://music.163.com/#/playlist?id=2045204807',
                 'http://music.163.com/#/playlist?id=2042006896',
                  'http://music.163.com/#/playlist?id=2055959723',
                  'http://music.163.com/#/playlist?id=2063351734',
                  'http://music.163.com/#/playlist?id=2049424729',
                  'http://music.163.com/#/playlist?id=2014095247',
                  'http://music.163.com/#/playlist?id=2023551932']
    gufeng_urls=['http://music.163.com/#/playlist?id=984166451',
                 'http://music.163.com/#/playlist?id=948471242',
                 'http://music.163.com/#/playlist?id=991478678',
                 'http://music.163.com/#/playlist?id=922791900']

    res_liuxing=Crawl(liuxing_urls,u'流行')
    res_gufeng=Crawl(gufeng_urls,u'古风')
    p = Pool(2)
    for i in range(2):
        if(i==0):
            p.apply_async(write_to_excel, args=(res_liuxing,'./data/liuxing.xls',))
        else:
            p.apply_async(write_to_excel, args=(res_gufeng, './data/gufeng.xls',))
    p.close()
    p.join()
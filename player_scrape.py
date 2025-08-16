import requests

from bs4 import BeautifulSoup

response = requests.get('https://www.covers.com/sport/football/nfl/teams/main/dallas-cowboys/roster')

print(response.status_code)

soup = BeautifulSoup(response.text, 'html.parser')

cookies = {
    'UserCountryCode': 'us',
    '_gid': 'GA1.2.1542449127.1755244951',
    '_gcl_au': '1.1.1655705391.1755244951',
    '_fbp': 'fb.1.1755244951015.163335582254065376',
    '_hjSession_1022261': 'eyJpZCI6ImQ5MmU3N2Y3LWE5YzctNDRlOS05OTdkLTY3NDY3MGE5NTU2YyIsImMiOjE3NTUyNDQ5NTEwODMsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjoxLCJzcCI6MX0=',
    '__qca': 'P1-3d2c774a-fd52-43ab-b4ab-47e893d00a40',
    '__hstc': '18899431.5b9cc70caf5872fa731000d8635df9c4.1755244951368.1755244951368.1755244951368.1',
    'hubspotutk': '5b9cc70caf5872fa731000d8635df9c4',
    '__hssrc': '1',
    'loggedIn': 'false',
    'kndctr_9CE579FD5DCD8B590A495E09_AdobeOrg_identity': 'CiY2NTM3NjExMzkwMzE3MTM2MTc5MzIxOTMzMzYwNzIxNzY1NjQwN1IQCO7wgeaKMxgBKgNPUjIwAfAB7vCB5ooz',
    'kndctr_9CE579FD5DCD8B590A495E09_AdobeOrg_cluster': 'or2',
    'AMCV_9CE579FD5DCD8B590A495E09%40AdobeOrg': 'MCMID|65376113903171361793219333607217656407',
    '_hjSessionUser_1022261': 'eyJpZCI6ImNhM2I1NWEyLTBkMmEtNWRmMS1hMmM1LTIyMjhmYjkwMDYwYSIsImNyZWF0ZWQiOjE3NTUyNDQ5NTEwODMsImV4aXN0aW5nIjp0cnVlfQ==',
    'CookieConsent': '{stamp:%27+yZyYWeAvJPaiaxvaR/rFdh6jIwY2YpMWGBw7kwEbg0cOBpDvB8wCA==%27%2Cnecessary:true%2Cpreferences:true%2Cstatistics:true%2Cmarketing:true%2Cmethod:%27explicit%27%2Cver:1%2Cutc:1755244978679%2Cregion:%27us-06%27}',
    'SSID_N': '0',
    'deeplink-modal': '2025-08-15T08%3A08%3A42.560Z',
    '_ga': 'GA1.2.117348307.1755244951',
    '_rdt_uuid': '1755244950909.87cdf84f-e5bc-4ab7-a05b-c77f7850b6a1',
    '_rdt_em': '0000000000000000000000000000000000000000000000000000000000000001',
    'adcloud': '{%22_les_v%22:%22c%2Cy%2Ccovers.com%2C1755249096%22}',
    '_ga_WEZ75VJ251': 'GS2.1.s1755244950$o1$g1$t1755247317$j60$l0$h0',
    '_gali': 'stats',
    '_v_id_l': '{%22_v_id%22:%229310468345109131755247296569%22%2C%22_la%22:1755247495728}',
}

headers = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'priority': 'u=1, i',
    'referer': 'https://www.covers.com/sport/football/nfl/teams/main/dallas-cowboys/stats',
    'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest',
    # 'cookie': 'UserCountryCode=us; _gid=GA1.2.1542449127.1755244951; _gcl_au=1.1.1655705391.1755244951; _fbp=fb.1.1755244951015.163335582254065376; _hjSession_1022261=eyJpZCI6ImQ5MmU3N2Y3LWE5YzctNDRlOS05OTdkLTY3NDY3MGE5NTU2YyIsImMiOjE3NTUyNDQ5NTEwODMsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjoxLCJzcCI6MX0=; __qca=P1-3d2c774a-fd52-43ab-b4ab-47e893d00a40; __hstc=18899431.5b9cc70caf5872fa731000d8635df9c4.1755244951368.1755244951368.1755244951368.1; hubspotutk=5b9cc70caf5872fa731000d8635df9c4; __hssrc=1; loggedIn=false; kndctr_9CE579FD5DCD8B590A495E09_AdobeOrg_identity=CiY2NTM3NjExMzkwMzE3MTM2MTc5MzIxOTMzMzYwNzIxNzY1NjQwN1IQCO7wgeaKMxgBKgNPUjIwAfAB7vCB5ooz; kndctr_9CE579FD5DCD8B590A495E09_AdobeOrg_cluster=or2; AMCV_9CE579FD5DCD8B590A495E09%40AdobeOrg=MCMID|65376113903171361793219333607217656407; _hjSessionUser_1022261=eyJpZCI6ImNhM2I1NWEyLTBkMmEtNWRmMS1hMmM1LTIyMjhmYjkwMDYwYSIsImNyZWF0ZWQiOjE3NTUyNDQ5NTEwODMsImV4aXN0aW5nIjp0cnVlfQ==; CookieConsent={stamp:%27+yZyYWeAvJPaiaxvaR/rFdh6jIwY2YpMWGBw7kwEbg0cOBpDvB8wCA==%27%2Cnecessary:true%2Cpreferences:true%2Cstatistics:true%2Cmarketing:true%2Cmethod:%27explicit%27%2Cver:1%2Cutc:1755244978679%2Cregion:%27us-06%27}; SSID_N=0; deeplink-modal=2025-08-15T08%3A08%3A42.560Z; _ga=GA1.2.117348307.1755244951; _rdt_uuid=1755244950909.87cdf84f-e5bc-4ab7-a05b-c77f7850b6a1; _rdt_em=0000000000000000000000000000000000000000000000000000000000000001; adcloud={%22_les_v%22:%22c%2Cy%2Ccovers.com%2C1755249096%22}; _ga_WEZ75VJ251=GS2.1.s1755244950$o1$g1$t1755247317$j60$l0$h0; _gali=stats; _v_id_l={%22_v_id%22:%229310468345109131755247296569%22%2C%22_la%22:1755247495728}',
}

params = {
    'teamId': '8',
    'seasonId': '590334',
    'leagueName': 'nfl',
    'countryCode': 'US',
    'stateProv': 'CA',
}

response = requests.get(
    'https://www.covers.com/sport/football/nfl/teams/main/dallas-cowboys/tab/roster',
    params=params,
    cookies=cookies,
    headers=headers,)

players = []

for table in soup.select("table.covers-CoversMatchups-Table"):
    rows = table.select("tbody tr")
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 5:
            continue
        name_tag = cols[0].find("a")
        href = name_tag['href']
        full_name_from_url = href.strip("/").split("/")[-1]  # extract last part of URL
        # replace hyphens with spaces and capitalize each word
        full_name_clean = full_name_from_url.replace("-", " ")

        player = {
            "full_name": full_name_clean,   # cleaned full name
            "number": cols[1].text.strip(),
            "height": cols[2].text.strip(),
            "weight": cols[3].text.strip(),
            "age": cols[4].text.strip()
        }
        players.append(player)



for p in players:
    print(p)




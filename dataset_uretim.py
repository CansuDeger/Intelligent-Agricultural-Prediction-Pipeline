import numpy as np
import pandas as pd
import random

np.random.seed(42)

rows = 100000

# TÜM TÜRKİYE İLLERİ
provinces = [
"Adana","Adıyaman","Afyonkarahisar","Ağrı","Amasya","Ankara","Antalya","Artvin",
"Aydın","Balıkesir","Bilecik","Bingöl","Bitlis","Bolu","Burdur","Bursa",
"Çanakkale","Çankırı","Çorum","Denizli","Diyarbakır","Edirne","Elazığ","Erzincan",
"Erzurum","Eskişehir","Gaziantep","Giresun","Gümüşhane","Hakkari","Hatay",
"Isparta","Mersin","İstanbul","İzmir","Kars","Kastamonu","Kayseri","Kırklareli",
"Kırşehir","Kocaeli","Konya","Kütahya","Malatya","Manisa","Kahramanmaraş",
"Mardin","Muğla","Muş","Nevşehir","Niğde","Ordu","Rize","Sakarya","Samsun",
"Siirt","Sinop","Sivas","Tekirdağ","Tokat","Trabzon","Tunceli","Şanlıurfa",
"Uşak","Van","Yozgat","Zonguldak","Aksaray","Bayburt","Karaman","Kırıkkale",
"Batman","Şırnak","Bartın","Ardahan","Iğdır","Yalova","Karabük","Kilis",
"Osmaniye","Düzce"
]

# BÖLGE BAZLI İKLİM
def climate_by_region(province):

    if province in ["Konya","Ankara","Kayseri","Sivas","Yozgat","Kırşehir","Nevşehir","Aksaray","Karaman"]:
        return 350,17,7

    elif province in ["Adana","Antalya","Mersin","Hatay","Osmaniye","Kahramanmaraş"]:
        return 650,24,7

    elif province in ["İzmir","Manisa","Aydın","Muğla","Denizli","Uşak"]:
        return 550,20,7

    elif province in ["Samsun","Trabzon","Rize","Ordu","Giresun","Sinop","Zonguldak","Bartın","Karabük"]:
        return 850,16,5

    elif province in ["Bursa","Balıkesir","Çanakkale","Edirne","Tekirdağ","Kırklareli","İstanbul","Kocaeli","Sakarya","Yalova","Bilecik"]:
        return 650,17,6

    elif province in ["Erzurum","Kars","Ağrı","Van","Muş","Bitlis","Hakkari","Ardahan","Iğdır"]:
        return 500,12,6

    else:
        return 300,26,8


crops = ["wheat","corn","potato"]

data = []

for i in range(rows):

    province = random.choice(provinces)
    crop = random.choice(crops)
    year = np.random.randint(2010,2024)

    base_rain,base_temp,base_solar = climate_by_region(province)

    rainfall = max(150, np.random.normal(base_rain,80))
    avg_temp = np.random.normal(base_temp,3)
    solar = np.random.normal(base_solar,0.7)

    humidity = np.random.uniform(40,80)
    daily_rain = np.random.uniform(0,20)
    wind = np.random.uniform(0,15)

    soil_n = np.random.uniform(20,80)
    soil_p = np.random.uniform(10,40)
    soil_k = np.random.uniform(80,200)

    soil_ph = np.random.uniform(6,8)
    soil_moist = np.random.uniform(10,40)
    soil_temp = avg_temp + np.random.normal(2,1)

    pesticides = np.random.uniform(1,8)
    gdd = np.random.uniform(1200,2000)
    plant_count = np.random.uniform(5,10)

    yield_ton = (
        rainfall*0.002 +
        gdd*0.002 +
        soil_n*0.02 +
        soil_p*0.015 +
        soil_k*0.01 +
        soil_moist*0.03 +
        solar*0.4 +
        pesticides*0.2 +
        plant_count*0.5 -
        abs(avg_temp-20)*0.1 +
        np.random.normal(0,0.5)
    )

    yield_ton = max(1, yield_ton)

    data.append([
        province,crop,year,rainfall,pesticides,avg_temp,
        humidity,daily_rain,wind,solar,
        soil_n,soil_p,soil_k,
        soil_moist,soil_ph,soil_temp,
        plant_count,gdd,yield_ton
    ])

columns = [
"Area","Item","Year","average_rainfall",
"pesticides","avg_temp","humidity",
"daily_rainfall","wind_speed","solar_radiation",
"soil_n","soil_p","soil_k",
"soil_moist","soil_ph","soil_temp",
"plant_count","GDD","yield_ton"
]

df = pd.DataFrame(data, columns=columns)

# 🔹 TÜRKÇE KARAKTER SORUNU ÇÖZÜMÜ
df["Area"] = df["Area"].str.normalize("NFKD").str.encode("utf-8", errors="ignore").str.decode("utf-8")

# =========================
#  5000'E DÜŞÜRME (SMART SAMPLING)
# =========================

TARGET_SIZE = 5000

groups = df.groupby(["Area","Item"])
num_groups = len(groups)

sample_per_group = TARGET_SIZE // num_groups

df_small_list = []

for (area,item), group in groups:

    if len(group) >= sample_per_group:
        df_small_list.append(group.sample(sample_per_group, random_state=42))
    else:
        df_small_list.append(group)

df_small = pd.concat(df_small_list)

remaining = TARGET_SIZE - len(df_small)

if remaining > 0:
    extra = df.drop(df_small.index).sample(remaining, random_state=42)
    df_small = pd.concat([df_small, extra])

df_small = df_small.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
#  KAYDET
# =========================

df_small.to_csv("turkey_agriculture_5k.csv", index=False, encoding="utf-8-sig")

print("Dataset hazır:", df_small.shape)
print("Şehir sayısı:", df_small["Area"].nunique())
print("Ürün dağılımı:\n", df_small["Item"].value_counts())
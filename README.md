# 🌾 Tarımsal Verim Tahmini için Makine Öğrenmesi Pipeline'ı

Türkiye genelindeki tarımsal verilerden yola çıkarak ürün verimini (`yield_ton`) tahmin eden, uçtan uca bir makine öğrenmesi pipeline'ı. İklim, toprak ve çevresel faktörleri kullanarak çoklu model karşılaştırması, otomatik hiperparametre optimizasyonu ve SHAP ile açıklanabilirlik analizi sunar.

## 🎯 Problem Tanımı

> *"Belirli bir bölge ve çevresel koşullar altında, bir tarım ürününün beklenen verimi ne olacaktır?"*

Bu, regresyon (sürekli değer tahmini) problemi olarak ele alınmıştır. Çalışma; **mısır (corn)**, **patates (potato)** ve **buğday (wheat)** için ayrı ayrı en iyi modeli belirlemeyi hedefler.

## 📊 Veri Seti

- **5.000 gözlem**, 19 değişken — eksik veri içermez
- İklim verileri: `average_rainfall`, `avg_temp`, `humidity`, `solar_radiation`, `wind_speed`
- Toprak özellikleri: `soil_n`, `soil_p`, `soil_k`, `soil_ph`, `soil_moist`, `soil_temp`
- Diğer: `pesticides`, `plant_count`, `GDD` (Growing Degree Days), `Year`, `Area`
- Hedef değişken: `yield_ton`
- Ürün dağılımı dengelidir (corn / potato / wheat birbirine yakın sayıda temsil edilir)

## 🛠️ Pipeline Akışı

1. **Veri Yükleme & Ön İnceleme** — yapısal analiz, istatistiksel özet
2. **Keşifsel Veri Analizi (EDA)** — ürün dağılımı, özellik histogramları
3. **Özellik Mühendisliği**
   - `rain_temp` = yağış × sıcaklık etkileşimi
   - `soil_total` = N + P + K toplamı
   - `climate_index` = nem × güneş radyasyonu
   - `year_trend` = yıl bazlı zamansal trend
4. **Modelleme** — Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
5. **Hiperparametre Optimizasyonu** — Optuna ile model başına 30 deneme
6. **Model Değerlendirme** — R², RMSE, MAE, MAPE + 5-fold cross-validation
7. **Açıklanabilirlik (XAI)** — Feature Importance + SHAP analizi
8. **Feature Selection** — önem skorlarına göre özellik sadeleştirme
9. **Final Model** — en iyi parametrelerle tüm veride yeniden eğitim, `.pkl` olarak kayıt

## 📈 Sonuçlar

Her üç üründe de en iyi performansı **CatBoost** göstermiştir:

| Ürün | En İyi Model | R² (Test) | RMSE | MAE | MAPE | CV R² (mean ± std) |
|---|---|---|---|---|---|---|
| Corn | CatBoost | 0.810 | 0.556 | 0.446 | %3.10 | 0.816 ± 0.020 |
| Potato | CatBoost | 0.806 | 0.559 | 0.437 | %3.02 | 0.821 ± 0.012 |
| Wheat | CatBoost | 0.812 | 0.577 | 0.467 | %3.22 | 0.825 ± 0.012 |

**Genel gözlemler:**
- CatBoost > XGBoost ≈ Gradient Boosting > Linear Regression > LightGBM > Random Forest
- Test ve cross-validation skorlarının birbirine yakın olması, modellerde **aşırı öğrenme (overfitting) olmadığını** gösterir
- Linear Regression'ın beklenenden iyi performans göstermesi, veri setinde güçlü lineer ilişkiler bulunduğuna işaret eder

## 🔍 Model Açıklanabilirliği (SHAP + Feature Importance)

Her üç üründe de tutarlı şekilde en etkili değişkenler:

1. `plant_count` — bitki yoğunluğu
2. `GDD` — büyüme derece günleri
3. `pesticides` — pestisit kullanımı
4. `soil_total` — toplam toprak besin değeri (N+P+K)

Düşük etkili değişkenler arasında `Year`, `humidity`, `Area`, `wind_speed` yer almaktadır.

**Feature Selection sonucu:** İlk 10-11 özellikle kurulan sadeleştirilmiş (reduced) modeller, tüm özelliklerin kullanıldığı modellerle neredeyse aynı (hatta bazı durumlarda marjinal olarak daha iyi) R² skoru üretmiştir — bu da modelin daha az değişkenle de verimli çalışabileceğini gösterir.

## 🧰 Kullanılan Teknolojiler

| Kategori | Araçlar |
|---|---|
| Veri işleme | pandas, numpy |
| Görselleştirme | matplotlib, seaborn |
| Modelleme | scikit-learn, XGBoost, LightGBM, CatBoost |
| Hiperparametre optimizasyonu | Optuna |
| Açıklanabilirlik | SHAP |
| Model kaydetme | joblib |

## 🚀 Kurulum ve Çalıştırma

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost optuna shap joblib
```

Notebook'u Jupyter veya benzeri bir ortamda açıp sırasıyla çalıştırın:

```bash
jupyter notebook Analiz_Model_Olusturma.ipynb
```

> ⚠️ **Not:** Veri seti şu anda sabit bir yoldan okunuyor:
> ```python
> df = pd.read_csv(r"C:\Users\DELL\Desktop\bitirme projem\model\turkey_agriculture_5k.csv", ...)
> ```
> Kendi ortamınızda çalıştırmadan önce bu satırı `turkey_agriculture_5k.csv` dosyanızın bulunduğu yola göre güncelleyin (göreceli bir yol kullanmanız önerilir, örn. `data/turkey_agriculture_5k.csv`).

## 📁 Çıktılar

Pipeline çalıştırıldığında her ürün için bir model dosyası üretilir:

```
final_model_corn.pkl
final_model_potato.pkl
final_model_wheat.pkl
```

Her `.pkl` dosyası, eğitilmiş modeli ve kullanılan nihai özellik listesini birlikte içerir:

```python
import joblib

saved = joblib.load("final_model_wheat.pkl")
model = saved["model"]
features = saved["features"]

prediction = model.predict(new_data[features])
```

## 📌 Notlar

- Bu proje bir bitirme/ders projesi kapsamında geliştirilmiştir
- Akademik çalışmalar, yarışmalar (Teknofest / TÜBİTAK) ve saha uygulamaları için temel oluşturacak şekilde tasarlanmıştır

## 🔮 Gelecek Çalışmalar

- Daha büyük ölçekli ve gerçek saha verileriyle model geliştirme
- Mobil/web tabanlı bir karar destek sistemine entegrasyon

---

**Geliştirici:** Cennet Cansu Değer

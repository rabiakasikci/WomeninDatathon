import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry_convert as pc
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from scipy import stats
from sklearn.model_selection import cross_val_score

def statistical_feature(data_stat, column_name):
    # Merkezi eğilim ölçüleri
    mean_values = data_stat[column_name].mean()
    median_values = data_stat[column_name].median()
    mode_values = data_stat[column_name].mode().iloc[0] if not data_stat[column_name].mode().empty else np.nan  # Mode boşsa np.nan döndür
    std_deviation = data_stat[column_name].std()
    variance = data_stat[column_name].var()
    quartiles = data_stat[column_name].quantile([0.25, 0.5, 0.75]).tolist()  # Listeye dönüştür

    # Sonuçları bir DataFrame'e dönüştür
    result_df = pd.DataFrame({
        "Özellikler": ["Merkezi Eğilim Ölçüleri", "Medyan Değerler", "Mod Değerleri", "Dağılım Ölçüleri", "Varyans", "Çeyreklikler"],
        "Değerler": [mean_values, median_values, mode_values, std_deviation, variance, quartiles]
    })
     
    

    return result_df

#Ülkelerin hangi kıtaya dahil olduğunu gösterir
#Burada Korenin code bilgisi null olduğu için dolduruldu.
def country_to_continent(country_name):
    try:
        if country_name=="Korea":
            country_name="North Korea"
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name
    except KeyError:
        #print(country_name)
        return "Unknown"

def fix_income_data(df, replacements):
    df_copy = df.copy()  # Veri çerçevesinin kopyasını oluşturuyoruz
    df_copy['Economy'] = df_copy['Economy'].replace(replacements)
    return df_copy
#Yapılan araştırma sonucu ülkelerin gelir seviyesi datası bulunmuş.
#data linki = https://datahelpdesk.worldbank.org/knowledgebase/articles/906519

income_df= pd.read_excel("../input/income-list/CLASS_yeni.xlsx")
replacements = {
    'Türkiye': 'Turkey',
    'Korea, Rep.': 'Korea',
    "Kyrgyz Republic" : "Kyrgyzstan"
}
updated_income_df = fix_income_data(income_df, replacements)

#income verisini datanın sonuna ekler
def merge_income_with_unpaid_care_work(data, income_df):
    income_list = []
    for i in range(len(data)):
        country_name = data.iloc[i][0]
        
        for j in range(len(income_df)):
            if country_name == income_df.iloc[j][0]:
                #print(country_name)
                income_list.append(str(income_df.iloc[j][1]))

    income_list_df = pd.DataFrame(income_list, columns=['Income'], index=data.index)
    return income_list_df

def plot_mean_by_country(dataframe, label_n):
    # Sayısal olmayan verileri temizle
    print("plot_mean_by_country")
    grouped = dataframe.groupby('Year')[label_n].mean()

# Grafik çizimi
    plt.figure(figsize=(12, 8))
    grouped.plot(marker='o', color='r')
    
    plt.title('Yıllara Göre Ortalama Kadın İşgücüne Katılım Oranı (%)')
    plt.xlabel('Yıl')
    plt.ylabel(label_n)
    plt.grid(False)
    plt.xticks()
    plt.tight_layout()
    plt.show()
    #plt.savefig("Yıllara Göre Ortalama Kadın İşgücüne Katılım Oranı (%)"+str(label_n))

    
    
    
    
    
    
    
   
#Bu fonksiyon sayesinde ilgili oranın yıllar içinde maximum ve minumun değerlerinin nasıl değiştiğini gösteren grafik çizebiliriz.
def plot_min_max_ratio(dataframe, column_name):
    print("plot_min_max_ratio")
    # Calculate minimum and maximum values for each entity
    min_max = dataframe.groupby("Entity")[column_name].agg(["min", "max"])

    # Find the rows corresponding to the minimum and maximum values for each entity
    min_rows = dataframe.loc[dataframe.groupby('Entity')[column_name].idxmin()]
    max_rows = dataframe.loc[dataframe.groupby('Entity')[column_name].idxmax()]

    # Merge the minimum and maximum rows on the 'Entity' column
    min_max_year = min_rows[['Entity', 'Year', column_name]].rename(columns={"Year": "Min Year", column_name: "Min Ratio"})
    min_max_year = min_max_year.merge(max_rows[['Entity', 'Year', column_name]].rename(columns={"Year": "Max Year", column_name: "Max Ratio"}),
                                      on="Entity")

    # Plotting
    plt.figure(figsize=(10, 6))  # Set the size of the plot

    # Scatter plot for minimum ratios
    plt.scatter(min_max_year["Min Year"], min_max_year["Min Ratio"], color='blue', label='Minimum Ratios', marker='o')

    # Scatter plot for maximum ratios
    plt.scatter(min_max_year["Max Year"], min_max_year["Max Ratio"], color='red', label='Maximum Ratios', marker='x')

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Ratio')
    plt.title('Distribution of Minimum and Maximum Ratios over Years for Entities')

    # Automatically adjust the axes
    plt.autoscale()

    # Show the legend, grid, and adjust layout
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #plt.savefig("Distribution of Minimum and Maximum Ratios over Years for Entities"+str(column_name))
    
def plot_value_by_continent(data, sutun):
    
    continent_df = pd.DataFrame(columns=['Continent'])

# Fill continent dataframe
    continent_list = []
    for i in range(len(data)):
        country_name = data.iloc[i, 0]
        continent_name = country_to_continent(country_name)
        #print(f"Country: {country_name}, Continent: {continent_name}")  # Debug print
        continent_list.append(continent_name)

    continent_df['Continent'] = continent_list

    result_df = pd.concat([data, continent_df], axis=1)
    average = result_df.groupby('Continent')[sutun].mean()
    average_df = pd.DataFrame({'Continent': average.index, 'Average Ratio': average.values})

    if 'Unknown' in average_df['Continent'].values:
    # Remove row with 'Unknown' continent
        average_df = average_df[average_df['Continent'] != 'Unknown']

# Plotting
    plt.figure(figsize=(10, 10))
    sns.barplot(data=average_df, 
                y='Continent', 
                x='Average Ratio', 
                palette='viridis')

# Plot customizations
    plt.title('Mean Value by Continent')
    plt.xlabel('Mean- ' + str(sutun))
    plt.ylabel('Continent ')
    plt.grid(True)
#plt.savefig('Mean Value by Continent.png', dpi=100, bbox_inches='tight')

# Show the plot
    plt.show()
    #plt.savefig("Mean Value by Continent"+str(sutun))
    
def plot_mmr(dataframe ,country_name, categories, label_n):
    filtered_df = dataframe[(dataframe['Entity'] == country_name) | (dataframe['Entity'].isin(categories))]
    grouped_df = filtered_df.groupby('Entity')

    plt.figure(figsize=(10, 10))
    for name, group in grouped_df:
        plt.plot(group['Year'], group[label_n], label=name)

    plt.xlabel('Year')
    plt.ylabel(label_n)
    plt.title(label_n + country_name)
    plt.legend()
    plt.grid(True)
    plt.show()
    #plt.savefig("plot_mmr"+str(label_n)+str(country_name))
    
   
def visualize_lfp_trends(dataframe, column_name, gender):
    # Get numeric columns for filling missing values and calculating mean
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    
    # Fill missing values with the mean of numeric columns
    dataframe.fillna(dataframe[numeric_columns].mean(), inplace=True)

    # İHD Sıralaması ile İşgücüne Katılma Oranı arasındaki ilişkiyi görselleştirme
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dataframe, x='HDI Rank (2021)', y=column_name)
    plt.title('İHD Sıralaması ve ' + column_name)
    plt.xlabel('İHD Sıralaması (2021)')
    plt.ylabel(column_name)
    plt.show()
    #plt.savefig("İHD Sıralaması ve"+str(column_name)+".png")
    

    # İşgücüne Katılma Oranı eğilimini inceleyelim
    plt.figure(figsize=(10, 10))
    relevant_columns = [col for col in dataframe.columns if 'Labour force participation rate, ' + gender in col]
    dataframe[relevant_columns].mean().plot()
    
    plt.title('İşgücüne Katılma Oranı Eğilimi (1990-2021)')
    plt.xlabel('Yıl')
    plt.ylabel(column_name)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #plt.savefig("İşgücüne Katılma Oranı Eğilimi (1990-2021)"+str(column_name)+".png")
    
def plot_lfp_by_country(df, label):
    # 'numeric_only' parametresini belirterek veri çerçevesini gruplayın
    grouped_df = df.groupby('Continent').mean(numeric_only=True)
    for country in grouped_df.index:
        years = [int(year.split('(')[-1].split(')')[0]) for year in grouped_df.columns[5:]]
        participation_rates = grouped_df.loc[country][5:]
        plt.plot(years, participation_rates, label=country, marker='o', linestyle='-')

    plt.xlabel('Years')
    plt.ylabel('Labour force participation rate (%)')
    plt.title('Labour force participation rate ' + label + ' by Country')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #plt.savefig("Labour force participation rate"+str(label)+' by Country.png')
    
def analyze_outliers(data):
    # Box plot ile aykırı değerleri görselleştirme
    plt.figure(figsize=(10, 6))
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    print(numeric_columns)
    plt.xticks(rotation=90)
    sns.boxplot(data=data[numeric_columns])
    plt.show()
    #plt.savefig("analyze_outliers"+str(data)+".png")
    

    # Scatter plot ile ilişkiyi görselleştirme


    # Z-skoru veya standart sapma yöntemi ile aykırı değerleri belirleme
    

    z_scores = stats.zscore(data[numeric_columns])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df_filtered = data[filtered_entries]
     
    # IQR yöntemi ile aykırı değerleri belirleme
    Q1 = data.quantile(0.25, numeric_only=True)
    Q3 = data.quantile(0.75, numeric_only=True)

    lower_bound = Q1 - 1.5 * (Q3 - Q1)
    upper_bound = Q3 + 1.5 * (Q3 - Q1)

    outliers = ((data.select_dtypes(include=np.number) < lower_bound) | (data.select_dtypes(include=np.number) > upper_bound)).any(axis=1)
    df_outliers = data[outliers]
    return df_outliers

def k_fold_cros(model, X,Y,k):
      # k değeri
    scores = cross_val_score(model, X, Y, cv=k, scoring='neg_mean_squared_error')

    # Negatif MSE'yi pozitif yapma ve ortalamasını alma
    mse_scores = -scores
    avg_mse = mse_scores.mean()

    print("K-Fold Cross-Validation MSE Scores:", mse_scores)
    print("Average MSE:", avg_mse)
	


##############################################################################################################
#                                        1. DATA İNCELEMESİ                                                  #
##############################################################################################################

#Veri Seti İnceleme & Açıklaması:
    
unpaid_care_work = pd.read_csv('../input/up-school-women-in-datathon-dataset/1- female-to-male-ratio-of-time-devoted-to-unpaid-care-work.csv')

print(unpaid_care_work.head())
print(unpaid_care_work.info())

"""
Veri setinde kadınların erkeklere göre ücretsiz bakım işlerine harcadıkları zamanın oranını incelemiştir
"""
# Deep Data (Datanın içine inme ve anlamlandırma)

"""
Hangi ülkelerde kadınlar erkeklere göre daha fazla ev işlerine zaman ayırdığını belirleyebiliriz. 
Bunu bir ortalama değerin üstünde olan ülkeler bazında bakabiliriz.
"""
mean_female_to_male_ratio = unpaid_care_work["Female to male ratio of time devoted to unpaid care work (OECD (2014))"].mean()
more_time_spent_by_women = unpaid_care_work[unpaid_care_work["Female to male ratio of time devoted to unpaid care work (OECD (2014))"] > mean_female_to_male_ratio]["Entity"]
print("Kadınların  erkeklere göre daha ortalamadan daha fazla ev işlerine zaman ayırdığı ülkeler:")
print(more_time_spent_by_women)


#Analiz Kısıtları

"""
Burada analizimizi kısıtlayacak null değerleri ortaya çıkarabiliriz.
Buna ek olarak income listesinde bazı ülkeler farklı bir adlandırma içeridği için fonksiyon üstünde o data düzeltilmiştir.
"""
missing_data = unpaid_care_work.isnull().sum()
print("Eksik veri sayısı:")
print(missing_data)


#Veri Analiz Çalışmasını yorumlama

"""
Burada öncelikle ülkelerin Female to male ratio of time devoted to unpaid care work değerlerini incelemek adına bir grafik çıkarımı yapılmıştır.
"""


plt.figure(figsize = (15,20))

ax=sns.barplot(data= unpaid_care_work,  
            y = 'Entity', 
            x = 'Female to male ratio of time devoted to unpaid care work (OECD (2014))')
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}', ((p.get_width() + 0.1), p.get_y() + p.get_height() / 2), ha='left', va='center')

plt.xticks(rotation=45)
plt.xlabel('Female to male ratio of time devoted to unpaid care work (OECD (2014))')
plt.ylabel('Country')

plt.title('Female to Male Ratio of Time Devoted to Unpaid Care Work by Country')

# Grafiği kaydetme
#plt.savefig('female_to_male_ratio_of_time_devoted_to_unpaid_care_work.png', dpi=100, bbox_inches='tight')

# Grafiği gösterme
plt.show()


"""
Bu bilgilerden yola çıkarak ülke bazında bu oranın nasıl olduğu incelenmek istendi.
"""

  

# Boş bir liste oluştur
continent_data = []

# Döngüyü çalıştır ve her döngü adımında listeye yeni bir sözlük ekle
for i in range(len(unpaid_care_work)):
    country_name = unpaid_care_work.iloc[i, 0]
    continent_name = country_to_continent(country_name)
    continent_data.append({'Entity': country_name, 'Continent': continent_name})
    

# continent_data listesini kullanarak bir DataFrame oluştur
continent_df = pd.DataFrame(continent_data)

# Ardından 'Continent' sütununu seçerek işlem yapın
result_df = pd.concat([unpaid_care_work, continent_df['Continent']], axis=1)


# Ardından gereken işlemleri yapabilirsiniz.
avarage = result_df.groupby('Continent')['Female to male ratio of time devoted to unpaid care work (OECD (2014))'].mean()
avarage_df = pd.DataFrame({'Continent': avarage.index, 'Average Ratio': avarage.values})

plt.figure(figsize=(10, 10))
sns.barplot(data=avarage_df, 
            y='Continent', 
            x='Average Ratio', 
            palette='viridis')

# Grafiği düzenlemeleri
plt.title('Mean- Female to male ratio of time devoted to unpaid care work (OECD (2014))')
plt.xlabel('Mean Value by Continent')
plt.ylabel('Continent ')
plt.grid(True)
#plt.savefig('Mean Value by Continent.png', dpi=100, bbox_inches='tight')

# Grafiği gösterelim
plt.show()
#plt.savefig("Mean- Female to male ratio of time devoted to unpaid care work (OECD (2014)")
    

plot_value_by_continent(unpaid_care_work, "Female to male ratio of time devoted to unpaid care work (OECD (2014))")
"""
Female to male ratio of time devoted to unpaid care work  oranının gelir seviyesi ile ilişkisini inceleyebilir miyiz sorusu üzerine
Yapılan araştırma sonucu ülkelerin gelir seviyesi datası bulunmuş birleştirilmiş ve ilişkisine bakılmıştır.
data linki = https://datahelpdesk.worldbank.org/knowledgebase/articles/906519
"""

income_list_df=merge_income_with_unpaid_care_work(unpaid_care_work, updated_income_df)

result_df = pd.concat([result_df, income_list_df], axis=1)

groupbyincome = result_df.groupby('Income')['Female to male ratio of time devoted to unpaid care work (OECD (2014))'].mean()

groupbyincome_df = pd.DataFrame({'Continent': groupbyincome.index, 'Average Ratio': groupbyincome.values})

plt.figure(figsize=(10, 10))
sns.barplot(data=groupbyincome_df, 
            y='Continent', 
            x='Average Ratio', 
            palette='inferno')

# Grafiği düzenlemeleri
plt.title('Income Level - Female/male ratio of time spent on unpaid care work (OECD (2014)) relationship')
plt.xlabel('Female to male ratio of time devoted to unpaid care work (OECD (2014))')
plt.ylabel('Income Level ')
plt.grid(True)
#plt.savefig('Income.png', dpi=100, bbox_inches='tight')

# Grafiği gösterelim
plt.show()

#############################################################################################################
#                                        2. DATA İNCELEMESİ                                                  #
##############################################################################################################

#Veri Seti İnceleme & Açıklaması:

share_of_wome_top_income = pd.read_csv('../input/up-school-women-in-datathon-dataset/2- share-of-women-in-top-income-groups.csv')


print(share_of_wome_top_income.head())
print(share_of_wome_top_income.info())

sns.pairplot(share_of_wome_top_income)
plt.show()
#plt.savefig("share_of_wome_top_income")
"""
Veri setinde zenginlik sıralamasındaki yüzdelik kısımlarda ne kadar kadın olduğunu gösteren bir bilgi vardır.
"""

# Deep Data (Datanın içine inme ve anlamlandırma)

"""
Veri setinde yer alan derin veriye bakarak, her bir ülke için zenginlik dağılımının nasıl değiştiğini ve cinsiyet eşitsizliği konusunda bilgiler edinebiliriz.
Örneğin, yıllara göre belirli zenginlik dilimlerindeki kadınların oranlarını karşılaştırabilir, eğilimleri belirleyebilir ve olası etmenleri değerlendirebiliriz.
"""

#Analiz Kısıtları

"""
Burada analizimizi kısıtlayacak null değerleri ortaya çıkarabiliriz.
UK datasını code kısmında null tespit edildi.
Buna ek olarak bazı ülkeler için bazı yüzdelikler için hiç bilgi yoktur.
Bu datalarda bundan sonraki analizde sıkıntı oluşturmaması adına çıakrılmıştır.
"""

missing_data = share_of_wome_top_income.isnull().sum()
share_of_wome_top_income['Code'] = share_of_wome_top_income['Code'].fillna("UK")
nan_columns = share_of_wome_top_income.columns[share_of_wome_top_income.isnull().all()].tolist()
country_df = share_of_wome_top_income.drop(columns=nan_columns)


#Veri Analiz Çalışmasını yorumlama

"""
Burada öncelikle ülke bazında bir analiz yapılmıştır. 
Kadınların yüzdelik dilime girme oraları yıllara göre data içindeki her ülke için ayrı ayrı grafikleştirilmiştir.
"""

# 'Entity' sütununda bulunan benzersiz değerleri alın
unique_values = share_of_wome_top_income['Entity'].unique()

# Her bir benzersiz değer (ülke) için döngü
for j in range(len(unique_values)):
    print(unique_values[j])
    
    # Ülkeye göre veri çerçevesini filtreleyin ve bir kopyasını oluşturun
    country_df = share_of_wome_top_income[share_of_wome_top_income['Entity'] == unique_values[j]].copy()
    
    # Eksik değerleri sadece sayısal sütunlarda ortalama ile doldurun
    numeric_columns = country_df.select_dtypes(include=['float64', 'int64']).columns
    country_df[numeric_columns] = country_df[numeric_columns].fillna(country_df[numeric_columns].mean())
    
    # Veriyi yıllara göre sıralayın
    country_df.sort_values(by='Year', inplace=True)
    
    # Grafik oluşturma işlemleri
    plt.figure(figsize=(10, 10))
    for column in numeric_columns[3:]:  # İlk üç sütunu atladık, çünkü bunlar 'Entity', 'Code' ve 'Year'
        x_values = country_df['Year']
        y_values = country_df[column]
        plt.plot(x_values, y_values, label=column)
        
    plt.xlabel('Years')
    plt.ylabel('Womens share in the rates')
    plt.title(str(unique_values[j]) + " - Womens share in the rates")
    plt.legend()
    plt.grid(True)
    #plt.savefig(str(unique_values[j]) + "_Shareofwomen.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    plt.close()

"""
Datanın istatiksel özelliri incelenmiştir.
"""
result_statistical_feature_01= statistical_feature(share_of_wome_top_income,'Share of women in top 0.1%')
result_statistical_feature_25= statistical_feature(share_of_wome_top_income,'Share of women in top 0.25%')
result_statistical_feature_5= statistical_feature(share_of_wome_top_income,'Share of women in top 0.5%')
result_statistical_feature_1= statistical_feature(share_of_wome_top_income,'Share of women in top 1%')
result_statistical_feature_10= statistical_feature(share_of_wome_top_income,'Share of women in top 10%')
result_statistical_feature_5= statistical_feature(share_of_wome_top_income,'Share of women in top 5%')
print(result_statistical_feature_01,result_statistical_feature_25,result_statistical_feature_5,
      result_statistical_feature_1,result_statistical_feature_10,result_statistical_feature_5 )

"""
Burada öncelikle ülke bazında bir analiz yapılmıştır. 
Kadınların yüzdelik dilime girme oraları yıllara göre her ülke için ayrı ayrı grafikleştirilmiştir.
"""


data_cleaned = share_of_wome_top_income.dropna(axis=1)
share_of_wome_top_income_cleaned = data_cleaned.columns

for i in share_of_wome_top_income_cleaned[3:]:
    #print(i)
    data_share_of_women = data_cleaned.pivot(index='Entity', columns='Year', values=i)
    
    # Yeni bir grafik oluşturma
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Grafik çizimini gerçekleştirme
    data_share_of_women.plot(ax=ax, kind='bar', legend=False,colormap='jet')
    
    # Grafik başlığını ve etiketlerini ayarlama
    ax.set_title(i)
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Country')
    
    # Grafikleri gösterme
    plt.tight_layout()
    plt.show()
    #plt.savefig(str(i)+"Percentage -Country ")
    
"""
Bu analizlere ek olarak maximum ve min değerlerin yıllar içinde nasıl değiştiği bulabiliriz. 
Bu sayede hangi yıllarda kadınların zenginlik oranında yer almasının maximum olduğunu bulabiliriz.
"""

plot_min_max_ratio(dataframe=share_of_wome_top_income, column_name="Share of women in top 10%")

plot_min_max_ratio(dataframe=share_of_wome_top_income, column_name="Share of women in top 5%")


"""
Elde edilen grafik ve sonuçlar incelendiğinde yıl geçtikçe kadınların en zengin oranlarında oranı artmıştır.
"""

##############################################################################################################
#                                        3. DATA İNCELEMESİ                                                  #
##############################################################################################################

#Veri Seti İnceleme & Açıklaması:



female_to_male_labor_force = pd.read_csv('../input/up-school-women-in-datathon-dataset/3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi.csv')

female_to_male_labor_force_stat= statistical_feature(female_to_male_labor_force,'Ratio of female to male labor force participation rate (%) (modeled ILO estimate)')

print(female_to_male_labor_force_stat)

"""
kadınların işgücüne katılma oranının erkeklerinkine göre ne kadar olduğunu gösteren bir data
"""
# Deep Data (Datanın içine inme ve anlamlandırma)

"""
düşük bir oran, kadınların işgücüne katılımında cinsiyet eşitsizliğinin veya ekonomik fırsatların eşit olmadığının bir göstergesi olabilir.
data içinde farklı gelir seviyesindeki değerlerde vardır. Bu yüzden bu değerler ile her ülkenin değeri kıyaslanabilir.
Bu sayede ülkelerin değerlerinin kıyaslaması yapılabilir.

"""


#Analiz Kısıtları

"""
Burada analizimizi kısıtlayacak null değerleri ortaya çıkarabiliriz.
Burada code değerlerinde bir eksiklik var bunu gidermek için aşağıdaki kod kullanılabilir.
"""
null_degerler = female_to_male_labor_force.isnull().sum()


for index, row in female_to_male_labor_force.iterrows():
    if pd.isnull(row['Code']):
        female_to_male_labor_force.at[index, 'Code'] = row['Entity'][:3]


null_degerler = female_to_male_labor_force.isnull().sum()

#Veri Analiz Çalışmasını yorumlama

"""
data içinde yılların tüm ülkeler bazında ortalaması alınarak yıllar içinde kadınların iş gücüne katılım oranın nasıl değiştiği bulabiliriz.
"""
plot_mean_by_country(female_to_male_labor_force,"Ratio of female to male labor force participation rate (%) (modeled ILO estimate)")
"""
Burada da görüldüğü üzere 1990 da 2020 ye kadar kadınların iş gücü artmıştır.
2020de bir düşüş yaşanmıştır. Bu değer pandemi ile alakalı olmuş olabilr.
"""

"""
Data bir ülke çin farklı yılardaki Kadınların erkeklere işgücüne katılım oranı verdiği için 
Max ve min değerlerin yıllara göre nasıl değiştiğini gösteren bir grafik çizebiliriz. 
Böylece yıllar için datanın max min değeri nasıl değiştiğini anlayabiliriz.
"""


plot_min_max_ratio(dataframe=female_to_male_labor_force, column_name="Ratio of female to male labor force participation rate (%) (modeled ILO estimate)")

income_entities_unique = female_to_male_labor_force[female_to_male_labor_force['Entity'].str.contains('income', case=False)]['Entity'].unique().tolist()



plot_mmr(female_to_male_labor_force,'Turkey', income_entities_unique,"Ratio of female to male labor force participation rate (%) (modeled ILO estimate)")


plot_mmr(female_to_male_labor_force,'Canada', income_entities_unique,"Ratio of female to male labor force participation rate (%) (modeled ILO estimate)")

"""
Burada da ülkelerin yüksek gelir ve düşük gelir ülkelerinde nasıl yer aldığını görebiliriz. Örneğin Türkiye için bir analiz yapacak olursak.
Kadın İşgücüne Katılım Oranı'nın Erkek İşgücüne Katılım Oranı farklı gelir seviyesine ait ülkelerin yıllar boyunca altında kalmıştır. 
Kanada ise bu konuda Türkiyenin zıttı bir eğilim izlemiştir.
"""

##############################################################################################################
#                                        5. DATA İNCELEMESİ                                                  #
##############################################################################################################

#Veri Seti İnceleme & Açıklaması:
maternal_mortality = pd.read_csv('../input/up-school-women-in-datathon-dataset/5- maternal-mortality.csv')
maternal_mortality_stat= statistical_feature(maternal_mortality,'Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))')

print(maternal_mortality.isnull().sum())

# Veri türlerini kontrol etme
print(maternal_mortality.dtypes)

print(maternal_mortality_stat)


#Analiz Kısıtları
"""
Veri setinde herhangi bir eksiklik yok gibi görünüyor. 
Ancak, analiz yaparken dikkate alınması gereken bazı kısıtlamalar olabilir. 
Örneğin, veri setindeki her bir gözlem, yalnızca belirli bir yıl için bir ülkeye aittir. 
Dolayısıyla, yıllar arasında veya ülkeler arasında doğrudan karşılaştırma yaparken dikkatli olmalıyız.

"""

#Veri Analizi
plot_value_by_continent(maternal_mortality, "Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))")
"""
Bu grafikten de ortaya çıkacağı üzere afrika kıtasında anne ölümleri çok fazladır.
Bunun sebebi sağlık koşulları olabilir.

"""



year_analysed = 2020
plt.figure(figsize=(20, 6))
df_2015 = maternal_mortality[maternal_mortality['Year'] == year_analysed]
plt.bar(df_2015['Entity'], df_2015['Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))'])
plt.xticks(rotation=90,fontsize=5)
plt.xlabel('Ülke')
plt.ylabel('Anne Mortalite Oranı (2015)')
plt.title('Ülkelerin'+str( year_analysed) +'Yılındaki Anne Mortalite Oranları')
plt.show()
#plt.savefig('Ülkelerin'+str( year_analysed) +'Yılındaki Anne Mortalite Oranları.png')
plot_mean_by_country(maternal_mortality,"Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))")

plot_min_max_ratio(dataframe=maternal_mortality, column_name="Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))")

"""
Burada da görüldüğü üzere 1990 da 2020 ye kadar kadınların iş gücü artmıştır.
2020de bir düşüş yaşanmıştır. Bu değer pandemi ile alakalı olmuş olabilr.
"""

income_entities_unique = maternal_mortality[maternal_mortality['Entity'].str.contains('income', case=False)]['Entity'].unique().tolist()
plot_mmr(maternal_mortality,'Turkey', income_entities_unique,"Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))")
plot_mmr(maternal_mortality,'Canada', income_entities_unique,"Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))")

"""
Burada da ülkelerin yüksek gelir ve düşük gelir ülkelerine kıyasla anne ölüm oranlarında nasıl yer aldığını görebiliriz.
Örneğin yine türkiye ve kanadayı karşılaştıracak olursak türkiye düşük oranını korumuş ve azaltmıştır.Kanada da yine bu oranda yılar içinde düşük bir seyir izlemiş
"""
plot_mmr(maternal_mortality,'Afghanistan', income_entities_unique,"Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))")

"""
Düşük bir oran beklediğimiz bir ülke olan Afganistana baktığımızda düşük gelirli ülkelerden bile fazla çıkmıştır.
"""

##############################################################################################################
#                                        6. DATA İNCELEMESİ                                                  #
##############################################################################################################

#Veri Seti İnceleme & Açıklaması:
"Belirli bir yılda cinsiyet arasındaki maaş farkının yüzdesini vermektedir"
gender_gap_in_average_wages = pd.read_csv('../input/up-school-women-in-datathon-dataset/6- gender-gap-in-average-wages-ilo.csv')
gender_gap_in_average_wages_stat= statistical_feature(gender_gap_in_average_wages,'Gender wage gap (%)')

print(gender_gap_in_average_wages_stat)
#Veri Kısıtları:
"""
Zaman Kısıtları: Veri seti belirli bir zaman aralığına ait olabilir ve daha geniş bir zaman aralığında analiz yapma isteği kısıtlanabilir. 
Eldeki veriler ile analiz yapmaya devam edildi.
"""

#Veri Analizi:

"""
Cinsiyetler arası ücret farkı, erkeklerin medyan kazançlarına kıyasla kadın ve erkeklerin medyan kazançları arasındaki fark olarak tanımlanmaktadır.
Negatif değer, kadınların medyan kazancının erkeklerinkinden daha yüksek olduğunu ifade eder. 
"""

biggerthanzero = gender_gap_in_average_wages[gender_gap_in_average_wages['Gender wage gap (%)'] > 0]
equalzero = gender_gap_in_average_wages[gender_gap_in_average_wages['Gender wage gap (%)'] == 0]
lessthanzero = gender_gap_in_average_wages[gender_gap_in_average_wages['Gender wage gap (%)'] < 0]
"""
Eğer bu fark sıfır (%0) ise, bu, kadınlar ve erkekler arasında ücretlerin eşit olduğu anlamına gelir.

Eğer gender wage gap (%) 0'dan büyükse, bu kadınların erkeklere kıyasla daha az kazandığı anlamına gelir.
"""
highest_countries = {}
lowest_countries = {}
for year in gender_gap_in_average_wages['Year'].unique():
    year_data = gender_gap_in_average_wages[gender_gap_in_average_wages['Year'] == year]
    max_growth_rate = year_data['Gender wage gap (%)'].max()
    min_growth_rate = year_data['Gender wage gap (%)'].min()
    highest_country = year_data[year_data['Gender wage gap (%)'] == max_growth_rate]['Entity'].values[0]
    lowest_country = year_data[year_data['Gender wage gap (%)'] == min_growth_rate]['Entity'].values[0]
    
    
    if highest_country != lowest_country:
        highest_countries[year] = {'Entity': highest_country, 'Gender wage gap (%)': max_growth_rate}
        lowest_countries[year] = {'Entity': lowest_country, 'Gender wage gap (%)': min_growth_rate}


highest_df = pd.DataFrame.from_dict(highest_countries, orient='index')
lowest_df = pd.DataFrame.from_dict(lowest_countries, orient='index')


plt.figure(figsize=(20, 10))

plt.bar(highest_df.index - 0.2, highest_df['Gender wage gap (%)'], width=0.4, color='green', label='Highest Growth Rate')
plt.bar(lowest_df.index + 0.2, lowest_df['Gender wage gap (%)'], width=0.4, color='red', label='Lowest Growth Rate')

plt.title('Highest and Lowest Gender wage gap (%) by Year')
plt.xlabel('Year')
plt.ylabel('Gender wage gap (%) (%)')
plt.xticks(gender_gap_in_average_wages['Year'].unique())
plt.legend()
plt.grid(axis='y')
for index, row in highest_df.iterrows():
    plt.text(index - 0.2, row['Gender wage gap (%)'] + 0.5, row['Entity'], ha='center', va='bottom', color='black', rotation=90)

for index, row in lowest_df.iterrows():
    plt.text(index + 0.2, row['Gender wage gap (%)'] + 0.5, row['Entity'], ha='center', va='bottom', color='black', rotation=90)

plt.tight_layout()
plt.show()
#plt.savefig("Highest and Lowest Gender wage gap (%) by Year".png)


plot_mean_by_country(gender_gap_in_average_wages,"Gender wage gap (%)")
plot_min_max_ratio(dataframe=gender_gap_in_average_wages, column_name="Gender wage gap (%)")

"""
Burada da görüldüğü üzere 1990 da 2020 ye kadar kadınların iş gücü artmıştır.
2020de bir düşüş yaşanmıştır. Bu değer pandemi ile alakalı olmuş olabilr.
"""

plot_value_by_continent(gender_gap_in_average_wages, "Gender wage gap (%)")

"""
Bu analiz ve görselde de yıllara göre en yüksek ve end düşüp gender wage gapi ve hangi ülkeye ait olduğunu inceleyebiliriz.
Burada en büyük fark 1993 yıında 2 farklı ülke içinde oluşmuştur.
"""

##############################################################################################################
#                                        7. DATA İNCELEMESİ                                                  #
##############################################################################################################

Labor_Force_Women_Entrpreneurship = pd.read_csv('../input/up-school-women-in-datathon-dataset/Labor Force-Women Entrpreneurship.csv',delimiter=";")

Labor_Force_Women_Entrpreneurship_stat= statistical_feature(Labor_Force_Women_Entrpreneurship,'Women Entrepreneurship Index')
#print(Labor_Force_Women_Entrpreneurship_stat)

#Veri Kısıtları
"""
Veri seti belirli bir coğrafi bölgeye veya döneme ait olabilir, bu da genellemeler yaparken dikkate alınmalı ve analizler ona göre yapılmalı.
"""

#Veri Analizleri

Labor_Force_Women_Entrpreneurship = Labor_Force_Women_Entrpreneurship.drop(columns=['Country'])
Labor_Force_Women_Entrpreneurship = Labor_Force_Women_Entrpreneurship.drop(columns=['No'])
Labor_Force_Women_Entrpreneurship['Level of development'] = pd.Categorical(Labor_Force_Women_Entrpreneurship['Level of development'])
Labor_Force_Women_Entrpreneurship['Level of development'] = Labor_Force_Women_Entrpreneurship['Level of development'].cat.codes

Labor_Force_Women_Entrpreneurship['European Union Membership'] = pd.Categorical(Labor_Force_Women_Entrpreneurship['European Union Membership'])
Labor_Force_Women_Entrpreneurship['European Union Membership'] = Labor_Force_Women_Entrpreneurship['European Union Membership'].cat.codes
Labor_Force_Women_Entrpreneurship['Currency'] = pd.Categorical(Labor_Force_Women_Entrpreneurship['Currency'])
Labor_Force_Women_Entrpreneurship['Currency'] = Labor_Force_Women_Entrpreneurship['Currency'].cat.codes
#print(Labor_Force_Women_Entrpreneurship)

correlation_matrix = Labor_Force_Women_Entrpreneurship.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Veri Seti Korelasyon Matrisi")
plt.show()
sns.pairplot(Labor_Force_Women_Entrpreneurship)
plt.show()

"Kadın Girişimcilik Endeksi ile Kadın İşgücüne Katılım Oranı arasındaki ilişkiyi ayrı bir şekilde de değerlendirebilirsiniz"


# Korelasyonu hesaplama
korelasyon= Labor_Force_Women_Entrpreneurship["Women Entrepreneurship Index"].corr(Labor_Force_Women_Entrpreneurship["Female Labor Force Participation Rate"])

print("Kadın Girişimcilik Endeksi ile Kadın İşgücüne Katılım Oranı arasındaki korelasyon:", korelasyon)


"""
korelasyon: 0.4413724223368963
Kadın Girişimcilik Endeksi ile Kadın İşgücüne Katılım Oranı arasında orta düzeyde pozitif bir ilişki olduğunu gösterir. 
Yani, bu iki değişken arasında belirli bir ilişki var, ancak ilişki çok güçlü değil. 
Değişkenler arasındaki ilişki, bir değişken artarken diğerinin de artma eğiliminde olduğunu, ancak bu ilişkinin güçlü olmadığını gösterir.
"""
korelasyon= Labor_Force_Women_Entrpreneurship["Women Entrepreneurship Index"].corr(Labor_Force_Women_Entrpreneurship["Inflation rate"])
print("Kadın Girişimcilik Endeksi ile enflasyon oranı arasındaki korelasyon:", korelasyon)

"""
Kadın Girişimcilik Endeksi ile enflasyon oranı arasındaki korelasyon: -0.45553237798193397
Kadın Girişimcilik Endeksi ile enflasyon oranı arasında orta düzeyde negatif bir ilişki olduğunu gösterir. 
Yani, bu iki değişken arasında bir ilişki var ve bu ilişki negatif yönlüdür. 
Bu, bir değişkenin artmasıyla diğer değişkenin azalma eğiliminde olduğunu ifade eder
"""

korelasyon= Labor_Force_Women_Entrpreneurship["Women Entrepreneurship Index"].corr(Labor_Force_Women_Entrpreneurship["Female Labor Force Participation Rate"])
print("Kadın Girişimcilik Endeksi ile Kadın İşgücüne Katılım Oranı arasındaki korelasyon:", korelasyon)


"""
Pozitif bir korelasyon, iki değişken arasında doğrusal bir ilişkinin olduğunu ve bu ilişkinin aynı yönde olduğunu gösterir. 
Yani, bir değişken artarken diğer değişkenin de artma eğiliminde olduğunu ifade eder.
Korelasyon katsayısı 0 ile 1 arasında olduğu için bu pozitif ilişkinin var olduğunu, ancak çok güçlü olmadığını gösterir.

"""
Labor_Force_Women_Entrpreneurship = pd.read_csv('../input/up-school-women-in-datathon-dataset/Labor Force-Women Entrpreneurship.csv',delimiter=";")

countries = Labor_Force_Women_Entrpreneurship['Country']

# Grafik çizimi
fig, ax = plt.subplots(figsize=(20, 6))

# Çizgi grafiği çizme
ax.plot(countries, Labor_Force_Women_Entrpreneurship['Women Entrepreneurship Index'], marker='o', label='Women Entrepreneurship Index')
ax.plot(countries, Labor_Force_Women_Entrpreneurship['Entrepreneurship Index'], marker='o', label='Entrepreneurship Index')
ax.plot(countries, Labor_Force_Women_Entrpreneurship['Inflation rate'], marker='o', label='Inflation rate')
ax.plot(countries, Labor_Force_Women_Entrpreneurship['Female Labor Force Participation Rate'], marker='o', label='Female Labor Force Participation Rate')

ax.set_ylabel('Değerler')
ax.set_xlabel('Ülkeler')
ax.set_title('Ülkelerin Farklı Özellikleri')
ax.legend()

plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()
#plt.savefig("Ülkelerin Farklı Özellikleri Labor_Force_Women_Entrpreneurship.png")

continent_data = []
for i in range(len(Labor_Force_Women_Entrpreneurship)):
    country_name = Labor_Force_Women_Entrpreneurship.iloc[i,1]
    continent_name = country_to_continent(country_name)
    continent_data.append({'Continent': continent_name})

continent_df = pd.DataFrame(continent_data)

result_df = pd.concat([Labor_Force_Women_Entrpreneurship, continent_df], axis=1)


df_outliers = analyze_outliers(Labor_Force_Women_Entrpreneurship)

##############################################################################################################
#                                        8. DATA İNCELEMESİ                                                  #
##############################################################################################################
#Veri Seti İnceleme & Açıklaması


Labour_Force_Participation_Male = pd.read_csv('../input/up-school-women-in-datathon-dataset/Labour Force Participation - Male.csv')

# Veri Analizi Çalışması ve Yorumlama
visualize_lfp_trends(Labour_Force_Participation_Male, 'Labour force participation rate, male (% ages 15 and older) (2021)', 'male')

numeric_columns = Labour_Force_Participation_Male.select_dtypes(include=np.number)
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Veri Seti Korelasyon Matrisi")
plt.show()

"Özellikler arasındaki ilişkileri görselleştir"

plt.figure(figsize=(20, 20))
sns.pairplot(Labour_Force_Participation_Male[['HDI Rank (2021)', 'Labour force participation rate, male (% ages 15 and older) (1992)']])

plt.show()

plot_lfp_by_country(Labour_Force_Participation_Male,"Male")
df_outliers = analyze_outliers(Labour_Force_Participation_Male)

##############################################################################################################
#                                        9. DATA İNCELEMESİ                                                  #
##############################################################################################################


#Veri Seti İnceleme & Açıklaması

Labour_Force_Participation_Female = pd.read_csv('../input/up-school-women-in-datathon-dataset/Labour Force Participation Female.csv')
# Veri Analizi Çalışması ve Yorumlama
visualize_lfp_trends(Labour_Force_Participation_Female, 'Labour force participation rate, female (% ages 15 and older) (2021)', 'female')

numeric_columns = Labour_Force_Participation_Female.select_dtypes(include=np.number)
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Veri Seti Korelasyon Matrisi")
plt.show()

sns.pairplot(Labour_Force_Participation_Female[['HDI Rank (2021)', 'Labour force participation rate, female (% ages 15 and older) (1992)']],corner=True)
plt.show()

plot_lfp_by_country(Labour_Force_Participation_Female,"Female")
df_outliers = analyze_outliers(Labour_Force_Participation_Female)


##############################################################################################################
#                                        10. DATA İNCELEMESİ                                                  #
##############################################################################################################


Placement = pd.read_csv('../input/up-school-women-in-datathon-dataset/Placement.csv')
sns.pairplot(Placement)
plt.show()

#Öncelikle işe girenlerin erkek ve kadın yüzdesine bakalım

placed_students = Placement[Placement["status"] == "Placed"]

# Cinsiyet dağılımını hesaplayın
gender_distribution = placed_students["gender"].value_counts()

# Pasta grafiği oluşturun
plt.figure(figsize=(8, 8))
plt.pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral'])
plt.title('İşe Yerleştirilen İnsanların Cinsiyet Dağılımı')
plt.show()
#plt.savefig("İşe Yerleştirilen İnsanların Cinsiyet Dağılımı.png")
"""
Burada da görüldüğü üzere erkeklerin işe girme yüzdeleri kadınların işe girme yüzdelerinden 2 kattan daha da fazla

"""


# Cinsiyet ve yüksek lisans durumuna göre gruplayın
grouped_data = placed_students.groupby(["gender", "specialisation"]).size().unstack(fill_value=0)

plt.figure(figsize=(8, 8))
labels = grouped_data.index
colors = ['skyblue', 'lightcoral']
for i, (label, color) in enumerate(zip(labels, colors)):
    plt.subplot(2, 2, i+1)
    plt.pie(grouped_data.loc[label], labels=grouped_data.columns, autopct='%1.1f%%', startangle=140, colors=[color, 'lightgrey'])
    plt.title(label)
plt.suptitle('İşe Yerleştirilen Öğrencilerin Cinsiyet ve Yüksek Lisans Durumuna Göre Dağılımı', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#plt.savefig("İşe Yerleştirilen Öğrencilerin Cinsiyet ve Yüksek Lisans Durumuna Göre Dağılımı.png")
"""
İşe giren yüzdelerine bakıldığında genellikle Mkt&Fin (Marketing & Finance - Pazarlama ve Finans) erkekler tercih edilirken
Mkt&HR (Marketing & HR - Pazarlama ve İnsan Kaynakları) kadınlar tercih edilmektedir.

"""


sorted_students = Placement.sort_values(by="emp_test_percentage", ascending=False)

slices = [10, 5, 15, 20]

# Dilimlerdeki öğrenci sayılarını hesaplayın ve cinsiyet dağılımlarını kaydedin
gender_distributions = {}
for slice_percent in slices:
    slice_size = int(len(Placement) * slice_percent / 100)
    slice_students = sorted_students.head(slice_size)
    gender_distributions[slice_percent] = slice_students["gender"].value_counts()

# Pasta grafiği oluşturun
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('İşe girme test sonuçlarına göre farklı Dilimlerdeki Öğrencilerin Cinsiyet Dağılımı', fontsize=16)

for i, (slice_percent, gender_distribution) in enumerate(gender_distributions.items()):
    ax = axs[i//2, i%2]
    ax.pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral'])
    ax.set_title(f'{slice_percent}% Dilim')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#plt.savefig("İşe girme test sonuçlarına göre farklı Dilimlerdeki Öğrencilerin Cinsiyet Dağılımı.png")


grouped_data = Placement.groupby(['gender', 'status', 'work_experience']).size().unstack(fill_value=0)



# Kadın ve erkek öğrencilerin yüzdeliklerini hesaplayın
gender_counts = Placement['gender'].value_counts()
gender_percentages = gender_counts / gender_counts.sum()


colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'lightgrey']

plt.figure(figsize=(12, 10))
for i, gender in enumerate(gender_percentages.index):
    for j, status in enumerate(['Placed', 'Not Placed']):
        label = f'{gender} {status}'
        percentages = [grouped_data.loc[(gender, status), exp] / grouped_data.loc[(gender, status)].sum() * 100 for exp in ['Yes', 'No']]
        plt.subplot(2, 2, i * 2 + j + 1)
        plt.pie(percentages, labels=['With Experience', 'No Experience'], autopct='%1.1f%%', colors=colors, startangle=140)
        plt.title(label)

plt.tight_layout()
plt.show()
#plt.savefig("gender_percentages.png")

correlation_matrix = Placement.corr(numeric_only=True)


# "ssc_percentage" ile "hsc_percentage" arasındaki korelasyonu alın
ssc_hsc_corr = correlation_matrix.loc["ssc_percentage", "hsc_percentage"]

print("ssc_percentage ile hsc_percentage arasındaki Pearson korelasyon katsayısı:", ssc_hsc_corr)

"""
Bu durumda, 0.511 olan korelasyon katsayısı, "ssc_percentage" ile "hsc_percentage" arasında orta düzeyde pozitif bir ilişki olduğunu gösterir. 
Yani, lise sınav yüzdesi ile yükseköğretim kurumu (hsc) sınav yüzdesi arasında bir ilişki vardır ve bir öğrencinin bir sınavda aldığı yüzde puanı arttığında, diğer sınavdaki yüzde puanının da artma eğiliminde olduğu söylenebilir. 
"""
df_outliers = analyze_outliers(Placement)

##############################################################################################################
#                                        MODEL GELİŞTİRME                                                    #
##############################################################################################################

"""
Elimizdeki datalar incelediği kullanılacak olanarı birleştirmek istendi. Eldeki datalar incelendiğinde gelecek veri tahmini olarak 
Labour force participation rate, female (% ages 15 and older) (2021) tahmin edilmeye çalışdı. 
Kadınların iş gücüne katıım oranını tahmin eden bir model oluşturulursa ilerli yıllarda kadınların iş gücünün ülke bazında nasıl bir oran izleneceği belirlebilir.
Bu yüzden eldeki veriler incelenerek eğitim datası haline getiriliştir. Labour force participation rate, female (% ages 15 and older) (2021) tahmin edimiştir.
"""

"""
Model geliştirmek için datayı birleştirmemiz gerekli
"""

share_of_women_top_income = pd.read_csv('../input/up-school-women-in-datathon-dataset/2- share-of-women-in-top-income-groups.csv')
female_to_male_labor_force= pd.read_csv('../input/up-school-women-in-datathon-dataset/3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi.csv')
maternal_mortality = pd.read_csv('../input/up-school-women-in-datathon-dataset/5- maternal-mortality.csv')
gender_gap_in_average_wages = pd.read_csv('../input/up-school-women-in-datathon-dataset/6- gender-gap-in-average-wages-ilo.csv')
Labor_Force_Women_Entrpreneurship = pd.read_csv('../input/up-school-women-in-datathon-dataset/Labor Force-Women Entrpreneurship.csv',  sep=';')
labour_force_participation_male = pd.read_csv('../input/up-school-women-in-datathon-dataset/Labour Force Participation - Male.csv')
labour_force_participation_female = pd.read_csv('../input/up-school-women-in-datathon-dataset/Labour Force Participation Female.csv')


# Veri çerçevelerinin "Entity" sütununu "Country" olarak yeniden adlandırma
Labor_Force_Women_Entrpreneurship.rename(columns={"Country": "Entity"}, inplace=True)
labour_force_participation_female.rename(columns={"Country": "Entity"}, inplace=True)
labour_force_participation_male.rename(columns={"Country": "Entity"}, inplace=True)
merged_df = (
    share_of_women_top_income
    .merge(female_to_male_labor_force, on=["Entity","Code" , "Year"], how="outer")
    .merge(maternal_mortality, on=["Entity","Code" , "Year"], how="outer")
    .merge(gender_gap_in_average_wages, on=["Entity", "Code" ,"Year"], how="outer")
    .merge(Labor_Force_Women_Entrpreneurship, on=["Entity"], how="outer")
    .merge(labour_force_participation_female, on=["Entity"], how="outer")
    .merge(labour_force_participation_male, on=["Entity"], how="outer")
)


# Yüzde 50'den az NaN içeren sütunları seçelim
merged_data = merged_df.dropna(axis=1, how='all')
nan_percentages = merged_data.isna().mean() * 100
selected_columns = nan_percentages[nan_percentages < 50].index
filtered_df = merged_data[selected_columns]
filtered_df = merged_data[selected_columns].copy()

#Kullanılmayacak sutunları silme
filtered_df.drop(['Code', 'ISO3_x', 'Hemisphere_x', 'Continent_y',	'Hemisphere_y',	'HDI Rank (2021)_y','ISO3_y'
 ], axis=1, inplace=True)


# Yeni bir özellik oluşturma: Kadın işgücü katılımındaki artış oranı

kadın_isgucu_sutunlari = [f"Labour force participation rate, female (% ages 15 and older) ({yil})" for yil in range(1990, 2022)]
filtered_df["KadinIsgucuArtisOrani"] = filtered_df[kadın_isgucu_sutunlari].diff(axis=1).mean(axis=1)

#correlation matrisi 

correlation_matrix = filtered_df.corr(numeric_only=True)
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Korelasyon Matrisi')
plt.show()
#plt.savefig("Korelasyon Matrisi_Model.png")


#outlier inceleme

df_outliers = analyze_outliers(correlation_matrix)
#kategorik verileri numerik yapma

label_encoder = LabelEncoder()

filtered_df['Continent_x'] = label_encoder.fit_transform(filtered_df['Continent_x'])

# Eksik değerleri medyan ile doldurma
for column in filtered_df.columns[1:]:
    median_value = filtered_df[column].median()
    # Using direct assignment
    filtered_df[column] = filtered_df[column].fillna(median_value)
    # Or using the suggested syntax
    # filtered_df.fillna({column: median_value}, inplace=True)
    
#Datayı test ve train ayarlama

X_columns = filtered_df.columns[1:37].append(filtered_df.columns[38:])
X = filtered_df[X_columns]

Y_column = "Labour force participation rate, female (% ages 15 and older) (2021)"
Y = filtered_df[Y_column]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


############################ MODEL OLUŞTURMA ############################

#Lineer Regression

model = LinearRegression()

k_fold_cros(model, X, Y,5)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


#modeli değerlendirme ve görselleştirme
plt.scatter(y_test, y_pred , alpha=0.5)
plt.xlabel("Gerçek Değer")
plt.ylabel("Tahmin Edilen Değer")
plt.title("Gerçek Değer vs. Tahmin Edilen Değer")
plt.show()
#plt.savefig("Gerçek Değer vs. Tahmin Edilen Değer_Model.png")

plt.hist(y_test - y_pred, bins=50)
plt.xlabel("Hata")
plt.ylabel("Frekans")
plt.title("Tahmin Hatalarının Dağılımı")
plt.show()
#plt.savefig("Tahmin Hatalarının Dağılımı_Model.png")




cv_scores = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')

# Negatif ortalama kare hata değerlerini pozitife çevirme
cv_scores = -cv_scores
mean_mse = np.mean(cv_scores)
std_mse = np.std(cv_scores)

# R^2 skoru hesaplama
cv_r2_scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
mean_r2 = np.mean(cv_r2_scores)
std_r2 = np.std(cv_r2_scores)

print("Cross-Validation Mean Squared Error:", mean_mse)
print("Cross-Validation MSE Standard Deviation:", std_mse)
print("Cross-Validation R^2 Score:", mean_r2)
print("Cross-Validation R^2 Score Standard Deviation:", std_r2)


"""
Ridge regresyonu, geleneksel lineer regresyon modeline bir düzenleme (regularization) yöntemi ekleyerek geliştirilmiş bir regresyon yöntemidir. Ridge regresyonu, aşırı uyum (overfitting) problemini azaltmaya ve modelin genelleme yeteneğini artırmaya yardımcı olur.
"""
#Ridge Modeli


ridge_model = Ridge(alpha=0.1)  # alpha: Regularizasyon katsayısı
ridge_model.fit(X_train, y_train)

# Eğitim ve test hatalarını hesaplayalım
train_preds = ridge_model.predict(X_train)
test_preds = ridge_model.predict(X_test)

train_mse = mean_squared_error(y_train, train_preds)
test_mse = mean_squared_error(y_test, test_preds)
ridge_coefficients = pd.Series(ridge_model.coef_, index=X.columns)
ridge_coefficients.plot(kind='bar')

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
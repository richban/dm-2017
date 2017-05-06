import pandas as pd



def reduce_dimensions(input_file_path):
    keep_columns = ['code', 'nutrition_grade_uk', 'nutrition_grade_fr', 'energy_100g', 'energy-from-fat_100g',
                    'fat_100g', 'saturated-fat_100g', '-butyric-acid_100g', '-caproic-acid_100g', '-caprylic-acid_100g',
                    '-capric-acid_100g', '-lauric-acid_100g', '-myristic-acid_100g', '-palmitic-acid_100g',
                    '-stearic-acid_100g', '-arachidic-acid_100g', '-behenic-acid_100g', '-lignoceric-acid_100g',
                    '-cerotic-acid_100g', '-montanic-acid_100g', '-melissic-acid_100g', 'monounsaturated-fat_100g',
                    'polyunsaturated-fat_100g', 'omega-3-fat_100g', '-alpha-linolenic-acid_100g',
                    '-eicosapentaenoic-acid_100g', '-docosahexaenoic-acid_100g', 'omega-6-fat_100g',
                    '-linoleic-acid_100g', '-arachidonic-acid_100g', '-gamma-linolenic-acid_100g',
                    '-dihomo-gamma-linolenic-acid_100g', 'omega-9-fat_100g', '-oleic-acid_100g', '-elaidic-acid_100g',
                    '-gondoic-acid_100g', '-mead-acid_100g', '-erucic-acid_100g', '-nervonic-acid_100g',
                    'trans-fat_100g', 'cholesterol_100g', 'carbohydrates_100g', 'sugars_100g', '-sucrose_100g',
                    '-glucose_100g', '-fructose_100g', '-lactose_100g', '-maltose_100g', '-maltodextrins_100g',
                    'starch_100g', 'polyols_100g', 'fiber_100g', 'proteins_100g', 'casein_100g', 'serum-proteins_100g',
                    'nucleotides_100g', 'salt_100g', 'sodium_100g', 'alcohol_100g', 'vitamin-a_100g',
                    'beta-carotene_100g', 'vitamin-d_100g', 'vitamin-e_100g', 'vitamin-k_100g', 'vitamin-c_100g',
                    'vitamin-b1_100g', 'vitamin-b2_100g', 'vitamin-pp_100g', 'vitamin-b6_100g', 'vitamin-b9_100g',
                    'vitamin-b12_100g', 'biotin_100g', 'pantothenic-acid_100g', 'silica_100g', 'bicarbonate_100g',
                    'potassium_100g', 'chloride_100g', 'calcium_100g', 'phosphorus_100g', 'iron_100g', 'magnesium_100g',
                    'zinc_100g', 'copper_100g', 'manganese_100g', 'fluoride_100g', 'selenium_100g', 'chromium_100g',
                    'molybdenum_100g', 'iodine_100g', 'caffeine_100g', 'taurine_100g', 'ph_100g',
                    'fruits-vegetables-nuts_100g', 'collagen-meat-protein-ratio_100g', 'cocoa_100g', 'chlorophyl_100g',
                    'carbon-footprint_100g', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g',
                    'glycemic-index_100g', 'water-hardness_100g']

    data_frame = pd.read_csv(input_file_path, delimiter="\t", usecols=keep_columns)

    data_frame = data_frame.dropna(subset=["nutrition-score-uk_100g"])  # filter out rows with no label
    data_frame = data_frame.dropna(how="all")  # filter rows with no data
    data_frame = data_frame.dropna(axis=1, how="all")  # filter columns with no values

    print("Reduced:")
    print(data_frame.count())

    return data_frame


def filter_sparse_rows(data_frame, threshold):
    data_frame = data_frame.dropna(thresh=round(len(data.columns) * threshold))  # filter out rows with fewer than x% observations

    data_frame = data_frame.dropna(how="all")  # filter rows with no data
    data_frame = data_frame.dropna(axis=1, how="all")  # filter columns with no values

    print("\nFiltered (" + str(threshold * 100) + "):")
    print(data_frame.count())

    return data_frame


path = "C:/Users/Tom/Documents/ITU/Studies/2017 Spring/Data Mining/Group Project/"
full_file = path + "openfoodfacts_full.csv"
kaggle_file = path + "openfoodfacts_kaggle.tsv"

data = reduce_dimensions(full_file)
#data.to_csv(path + "reduced.tsv", sep="\t")  # output reduced file to a new file

thresholds = [.1, .2, .3, .4, .5]

for t in thresholds:
    filtered = filter_sparse_rows(data, t)
    filtered.to_csv(path + "reduced_filtered_" + str(int(t * 100)) + ".tsv", sep="\t")

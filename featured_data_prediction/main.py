from data_utils import process_sources, epoching, extract_features, load_featured_data
from model_utils import evaluate_10_folds
from tqdm import tqdm
itemNums = [*range(507, 544), *range(545, 571), *range(573, 629)]
valid_events = ['1', '2', '3', '4', '5', '6', '11', '12', '13', '14', '15', '16']

matPathBase = './Data/mat_files/'
setPathBase = './Data/set_files/'
epochPathBase = './Data/epoched_data/'
featPathBase = './Data/extracted_features/'
labelPath = './Data/labels.xlsx'

CREATE_SET_FILES = True
DO_EPOCHING = True
EXTRACT_FEATURES = True

if CREATE_SET_FILES:
	print('Processing Source (.mat) files and converting them to .set files')
	for item in tqdm(itemNums):
		process_sources(item, matPathBase, setPathBase)

if DO_EPOCHING:
	print('Epoching data based on events')
	for item in tqdm(itemNums):
		epoching(item, setPathBase, epochPathBase, valid_events)

if EXTRACT_FEATURES:
	print('Extract features')
	for item in tqdm(itemNums):
		extract_features(item, epochPathBase, featPathBase)

X, Y = load_featured_data(featPathBase, labelPath, itemNums)
evaluate_10_folds(X, Y)
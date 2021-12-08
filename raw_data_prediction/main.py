from data_utils import process_sources, epoching, extract_features, load_featured_data
from model_utils import evaluate_10_folds
from tqdm import tqdm
itemNums = [*range(507, 544), *range(545, 571), *range(573, 629)]
valid_events = ['1', '2', '3', '4', '5', '6', '11', '12', '13', '14', '15', '16']

matPathBase = './Data/mat_files/'
setPathBase = './Data/set_files/'
labelPath = './Data/labels.xlsx'

CREATE_SET_FILES = True
EVENT_TYPE = 'open_eyes'
OVERLAP_RATE = 0.9
WINDOW_TIME = 5


if CREATE_SET_FILES:
		if EVENT_TYPE=='open_eyes':
				events = ['2', '4', '6', '12', '14', '16']
		elif EVENT_TYPE=='close_eyes':
				events = ['1', '3', '5', '11', '13', '15']
    else:
        raise('undefined event type!')
		print('Processing Source (.mat) files, get specified events and converting them to .set files')
		for item in tqdm(itemNums):
				prepare_sources(item, matPathBase, setPathBase, events)

X, Y = preprocess_data(itemNums, setPathBase, labelPath, OVERLAP_RATE, WINDOW_TIME)

evaluate_10_folds(X, Y)
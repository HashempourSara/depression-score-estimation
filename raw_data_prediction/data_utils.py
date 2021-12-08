import numpy as np
import pandas as pd
import scipy
from oct2py import octave
import mne
import h5py


def events_seperation(matData, events):
		events = matData['event']['type'][0]
		latency = matData['event']['latency'][0].astype(int)
		data = matData['data']
		seperated_data = np.zeros((data.shape[0], 1))
		maxI = events.shape[0]
		for i in range(maxI):
				if (events[i] in events) or (events[i] in np.array(events).astype(int)):
						tmin = latency[i]-1
						if i == maxI-1:
								tmax = data.shape[1]
						else:
								tmax = latency[i+1]-1
						seperated_data = np.append(seperated_data, data[:, tmin:tmax], axis=1)
		seperated_data = np.delete(seperated_data, 0, 1)
		return seperated_data


def prepare_sources(item, matPathBase, setPathBase, events):
		matName = f'{item}_Depression_REST'
		matItem = matPathBase + matName + '.mat'
		matData = octave.pop_loadset(matItem)
		seperated_data = events_seperation(matData, events)
		matData['times'] = np.empty(0)
		matData['urevent'] = np.empty(0)
		matData['event'] = np.empty(0)

		setItem = setPathBase + f'{item}.set'
		matData['data'] = seperated_data
		matData['pnts'] = float(seperated_data.shape[1])
		temp = octave.eeg_checkset(matData)
		octave.pop_saveset(temp, filename=str(item)+'.set', check='on',
											filepath=setPathBase, savemode='onefile');


def process_signal(raw):
		## Set reference
		raw.set_eeg_reference(ref_channels=['M1', 'M2'])

		## Drop Bad Channels
		raw.drop_channels([x for x in ['CB1', 'CB2', 'EKG', 'HEOG', 'VEOG'] if x in raw.ch_names])

		## Remove Baseline
		raw._data = mne.baseline.rescale(raw.get_data(), raw.times, (None, None))

		## Apply notch filter
		raw.notch_filter(50)

		## Apply Frequency Band Filter
		raw.filter(0.2, 45, method='iir', verbose='WARNING')

		## Apply Butter Filter
		lowcut = 1.0
		highcut = 50.0
		order = 5
		nyq = 0.5 * raw.info['sfreq']
		low = lowcut / nyq
		high = highcut / nyq
		b, a = scipy.signal.butter(order, [low, high], btype='band')
		raw._data = scipy.signal.lfilter(b, a, raw.get_data())

		## Apply ICA Decomposition
		ica = mne.preprocessing.ICA(max_iter=100, random_state=1)
		ica.fit(raw)
		ica.apply(raw)
		return raw


def windowing(x, y, win_size, bias):
		windowed_x = []
		window_labels = []
		for i in range(0, x.shape[0], bias):
				temp = x[i:i+win_size, :]
				if temp.shape[0]==win_size:
						windowed_x.append(temp)
						window_labels.append(y)
		return windowed_x, window_labels


def pad_N_epochs(x,limit):
		out = []
		for y in x:
				while len(y) < limit:
						y = np.concatenate([y,y[-(limit-len(y)):]])
				out.append(y[:limit])
		return np.array(out)

def crop_N_epochs(x,limit):
		return np.array([y[:limit] for y in pad_N_epochs(x,limit)])


def preprocess_data(itemNums, setPathBase, labelPath, olp, window_time):
		fs = 250
		win_size = window_time*fs
		bias = int((1.0-olp)*win_size)
		seq_len = 60000
		# load labels file
		labels_data = pd.ExcelFile(labelPath)
		labels_data = labels_data.parse('Depression Rest')

		idx = np.array(range(0,seq_len,2))
		data = []
		labels = []
		for item in itemNums:
				setItem = setPathBase + f'{item}.set'
				raw = mne.io.read_raw_eeglab(setItem)
				raw = process_signal(raw)

				item_data = crop_N_epochs(raw._data, seq_len)
				item_data = item_data[:, idx]
				item_data = item_data.reshape(item_data.shape[0], item_data.shape[1])

				item_data = item_data.transpose([1, 0])
				item_label = labels_data[labels_data['id']==item].BDI.values[0]
				windowed_item_data, window_labels = windowing(item_data, item_label, win_size, bias)
				data.append(windowed_item_data)
				labels.append(window_labels)

		data = np.concatenate(data)	
		labels = np.concatenate(labels).astype('int16').reshape(-1, 1)
		return data, labels
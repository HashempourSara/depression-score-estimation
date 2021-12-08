import numpy as np
import pandas as pd
import scipy
from oct2py import octave
import mne
import pywt

def list2arr(*argv):
	out = np.empty(len(argv),dtype=np.object)
	for i in range(len(argv)):
		out[i] = argv[i]
	return out

def process_sources(item, matPathBase, setPathBase):
	matName = f'{item}_Depression_REST'
	matItem = matPathBase + matName + '.mat'
	# load mat filr
	matData = octave.pop_loadset(matItem)
	matData = octave.eeg_checkset(matData)
	# save data as set file
	octave.pop_saveset(matData, filename=f'{item}.set', 
						filepath=setPathBase, savemode='onefile', 
						check='on');


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

def epoching(item, setPathBase, epochPathBase, valid_events):
	setItem = setPathBase + f'{item}' + '.set'
	fepoch = epochPathBase + f'{item}' + '_epo'
	ftimes = epochPathBase + f'{item}' + '_ept'
	## Read SET file
	raw = mne.io.read_raw_eeglab(setItem)
	## Fetch Events
	events, event_id = mne.events_from_annotations(raw)
	event_id = dict((k.replace('.0',''),v) for k,v in event_id.items())
	event_id_swapd = dict((v,k) for k,v in event_id.items())
	## Process signal
	raw = process_signal(raw)
	## Apply Epoching
	epochs = {}
	times = {}
	maxI = len(events) - 1
	prev_ev_name = None
	for i,ev in enumerate(events):
		ev_name = event_id_swapd[ev[2]]
		if ev_name not in valid_events:
			continue
		tmin = ev[0]
		if i == maxI:
			tmax = raw._data.shape[1]
		else:
			tmax = events[i+1][0]
			if len(events) > i+2 and ev[-1] not in [14,15] and events[i+1][-1] in [14,15]:
				ccc,j = 0,1
				while( events[i+j][-1] in [14,15] ): ccc+=1;j+=1;
				last = events[i+j]
				if (ev[-1] in [1,9,10,11,12,13]
					and (last[-1] == ev[-1] or last[0]-tmin <= 270)
					or (last[-1] != ev[-1] and tmax-tmin < 174 and last[0]-tmin <= 176)
					):
					tmax = last[0]
		dd = raw._data[:,tmin:tmax].transpose()
		tt = raw.times[tmin:tmax]
		if epochs.get(ev_name) is None:
			epochs[ev_name] = [[dd]]
			times[ev_name] = [[tt]]
		else:
			if prev_ev_name == ev_name:
				epochs[ev_name][-1].append(dd)
				times[ev_name][-1].append(tt)
			else:
				epochs[ev_name].append([dd])
				times[ev_name].append([tt])
		prev_ev_name = ev_name
	if len(epochs) < len(valid_events):
		print('Subject does not have all events:',item)
		return False
	epochsNp = []
	timesNp = []
	for ev in valid_events:
		epochsNp.append(np.array([list2arr(*x) for x in epochs[ev]]))
		timesNp.append(np.array([list2arr(*x) for x in times[ev]]))
	epochsNp = list2arr(*epochsNp)
	timesNp = list2arr(*timesNp)
	np.save(fepoch, epochsNp)
	np.save(ftimes, timesNp)
	return True

def features_of_channel(data, timesdiff):
	if len(data) < 3:
		return None
	sfreq = 500
	dwt='db4'
	# DWT - Discrete Wavelet Transform
	cA,cD = pywt.dwt(data, dwt)
	# DWT - FFT of Approximation
	cAf = np.fft.fft(cA)
	# DWT - FFT of Details
	cDf = np.fft.fft(cD)
	# Diff - First Order Difference
	diff1 = np.diff(data)
	# Diff - Second Order Difference
	diff2 = np.diff(data,n=2)
	# Diff - Variance of First Order Difference
	diff1Var = np.mean(diff1 ** 2)
	# Hjorth - Activity
	activity = np.var(data)
	# Hjorth - Mobility
	mobility = np.sqrt(diff1Var / activity)
	# Hjorth - Complexity
	complexity = np.sqrt(np.mean(diff2 ** 2) / diff1Var) / mobility
	# FFT - Fast Fourier Transform
	fft1 = np.fft.fft(data)
	# FFT - ABS(FFT)
	fft1abs = np.abs(fft1.real)
	# Bands - Delta/Theta/Alpha/Beta
	bands = {
	  'delta':{'freq':(0.5,4),'sum': 0,'max': 0},
	  'theta':{'freq':(4,8)  ,'sum': 0,'max': 0},
	  'alpha':{'freq':(8,13) ,'sum': 0,'max': 0},
	  'beta' :{'freq':(13,30),'sum': 0,'max': 0},
	}
	shape1sfreq = float(data.shape[0])/sfreq
	for band,bandDict in bands.items():
		arange = np.arange(bandDict['freq'][0]*shape1sfreq,
							bandDict['freq'][1]*shape1sfreq, dtype=int)
		bandDict['sum'] = np.sum(fft1abs.real[arange])
		bandDict['max'] = np.max(fft1abs.real[arange])
	# Vertex to Vertex Slope
	diff1Slope = diff1/timesdiff
	result = np.array([
						np.min(data),                                #   'Min'
						np.max(data),                                #   'Max'
						np.std(data),                                #   'STD'
						np.mean(data),                               #   'Mean'
						np.median(data),                             #   'Median'
						activity,                                    #   'Activity'
						mobility,                                    #   'Mobility'
						complexity,                                  #   'Complexity'
						scipy.stats.kurtosis(data),                  #   'Kurtosis'
						np.mean(diff2),                              #   '2nd Difference Mean'
						np.max(diff2),                               #   '2nd Difference Max'
						np.mean(diff1),                              #   '1st Difference Mean'
						np.max(diff1),                               #   '1st Difference Max'
						scipy.stats.variation(data),                 #   'Coeffiecient of Variation'
						scipy.stats.skew(data),                      #   'Skewness'
						np.mean(cA),                                 #   'Wavelet Approximate Mean'
						np.std(cA),                                  #   'Wavelet Approximate Std Deviation'
						np.mean(cD),                                 #   'Wavelet Detailed Mean'
						np.std(cD),                                  #   'Wavelet Detailed Std Deviation'
						np.sum(np.abs(cAf) ** 2) / cAf.size,         #   'Wavelet Approximate Energy'
						np.sum(np.abs(cDf) ** 2) / cDf.size,         #   'Wavelet Detailed Energy'
						-np.sum(cA * np.nan_to_num(np.log(cA))),     #   'Wavelet Approximate Entropy'
						-np.sum(cD * np.nan_to_num(np.log(cD))),     #   'Wavelet Detailed Entropy'
						np.mean(diff1Slope),                         #   'Mean of Vertex to Vertex Slope'
						np.var(diff1Slope),                          #   'Var  of Vertex to Vertex Slope'
						bands['delta']['max'],                       #   'FFT Delta Max Power'
						bands['theta']['max'],                       #   'FFT Theta Max Power'
						bands['alpha']['max'],                       #   'FFT Alpha Max Power'
						bands['beta' ]['max'],                       #   'FFT Beta Max Power'
						bands['delta']['sum']/bands['alpha']['sum'], #   'Delta/Alpha'
						bands['delta']['sum']/bands['theta']['sum'], #   'Delta/Theta'
						])
	return result


def extract_features(item, epochPathBase, featPathBase):
	fepoch = epochPathBase + f'{item}' + '_epo.npy'
	ftimes = epochPathBase + f'{item}' + '_ept.npy'
	featPath = featPathBase + f'{item}' + '_feat'
	epochs = np.load(fepoch, allow_pickle=True)
	times  = np.load(ftimes, allow_pickle=True)
	ft = []
	# per event
	for ev in range(len(epochs)):
		# per epochs sequence
		ft.append([])
		for epo_gr in range(len(epochs[ev])):
			# per single epoch
			ft[-1].append([])
			for epo in range(len(epochs[ev][epo_gr])):
				# per channel
				timesdiff = np.diff(times[ev][epo_gr][epo])
				ft[-1][-1].append([])
				for ch in range(epochs[ev][epo_gr][epo].shape[1]):
					res = features_of_channel(epochs[ev][epo_gr][epo][:,ch], timesdiff)
					if res is not None:
						ft[-1][-1][-1].append(res)
					if not len(ft[-1][-1][-1]):
						del ft[-1][-1][-1]
				ft[-1][-1][-1] = np.array(ft[-1][-1][-1])
			ft[-1][-1] = np.array(ft[-1][-1])
		ft[-1] = np.concatenate(ft[-1])
	ft = list2arr(*ft)
	np.save(featPath, ft)
	return True


def pad_N_epochs(x,limit):
	out = []
	for y in x:
		while len(y) < limit:
			y = np.concatenate([y,y[-(limit-len(y)):]])
		out.append(y[:limit])
	return np.array(out)


def load_featured_data(featPathBase, labelPath, itemNums):
	# load labels file
	labels_data = pd.ExcelFile(labelPath)
	labels_data = labels_data.parse('Depression Rest')
	# load feature data
	data = []
	labels = []
	counter = 0
	for item in itemNums:
		counter += 1
		featPath = featPathBase + f'{item}' + '_feat.npy'
		temp_data = np.load(featPath, allow_pickle=True)
		temp_data = pad_N_epochs(temp_data, 120)
		temp_data = temp_data.reshape(temp_data.shape[0], temp_data.shape[1], -1)
		data.append(temp_data)
		lab = labels_data[labels_data['id']==item].BDI.values[0]*np.ones(temp_data.shape[0])
		labels.append(lab)

	data = np.concatenate(data)
	labels = np.concatenate(labels).astype('uint8')
	return data, labels
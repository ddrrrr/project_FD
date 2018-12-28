# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:53 2018

@author: a273
TODO
    should class DataSet only arange, save and load?
"""

import os
import operator
import random
import scipy.io as sio
import pickle as pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
import shutil

class DataSet(object):
    '''This class is used to arrange dataset, collected and used by Lab 119 in HIT.
        module numpy, pickle and pandas should be installed before used.
        Attributes:
            name: The name of dataset with str type. And this name is used to save and load the 
                dataset with the file name as 'DataSet_' + name + '.pkl'
            info: An OrderedDict contained all the attribution of dataset.
            save_path: A string described where to save or load this dataset, and defaulted as './data/'
            dataset: A list contained samples and their attributes.
            save_in_piece: A bool to decide if save dataset in piece
    '''
    def __init__(self,name='',info=OrderedDict(),save_path='./data/',dataset=[],len_data=0,load_name=''):
        self.name = name
        self.info = info
        self.save_path = save_path
        self.dataset = dataset
        self.len_data = len_data
        self.load_name = load_name
        self.save_in_piece = False

    # inner function
    def _deal_condition(self,condition):
        '''
        get the index of samples whose attributes is in condition.

        Args:
            condition: A dict whose keys are the name of attributes and values are lists contained values owned by 
                samples we need.
        Return:
            conforming_idx: A list contained index of samples whose attributes is in condition.
        '''

        idx_all = []
        for k in condition.keys():
            idx_all.append(set([i for i,x in enumerate(self.info[k]) if x in condition[k]]))
        return list(set.intersection(*idx_all))

    # modify
    def add_index(self,new_attribute,new_value=None):
        '''
        Add new attribute to dataset.
        
        Args:
            new_attribute: The name of new attribute (a string).
            new_value: A list contained values appended to each sample. If the length of new_value is 1,
                then all samples will append the same new_value. Or the length of new_value should be the
                same as the number of samples, then each sample will append the corresponding value. 
                Otherwise, raise valueError.
        Return:
            None
        '''
        if new_value == None:
            self.info[new_attribute] = [None]*self.len_data
        elif isinstance(new_value,list):
            if len(new_value) == 1:
                self.info[new_attribute] = new_value*self.len_data
            elif len(new_value) == self.len_data:
                self.info[new_attribute] = new_value
            else:
                raise TypeError

    def del_index(self,del_attribute):
        '''
        delete attribute and the corresponding values in each sample.
        
        Args:
            del_attribute: The name of attribute (a string).
        Return:
            None
        '''
        del self.info[del_attribute]

    def append(self,append_data):
        '''
        Append one sample.
        
        Args:
            append_data: A dict or a list that contain a sample, including data and attribute.
                If append_data is a list, then data should be the first element, and the other elements should be in order.
        Return:
            None
        '''
        if isinstance(append_data,dict):
            if len(append_data.keys()) <= len(self.info.keys())+1:
                for k in append_data.keys():
                    if k == 'data':
                        self.dataset.append(append_data[k])
                    elif k not in self.info.keys():
                        self.info[k] = [None]*(self.len_data+1)
                    else:
                        self.info[k].append(append_data[k])
                self.len_data += 1
            else:
                raise ValueError('append_data has too much attribute!')
        elif isinstance(append_data,list):
            if len(append_data) == len(self.info.keys())+1:
                self.dataset.append(append_data.pop(0))
                for k in self.info.keys():
                    self.info[k].append(append_data.pop(0))
                self.len_data += 1
            else:
                raise ValueError('append_data has wrong number of attribute!')
        else:
            raise TypeError('append_data should be dict or list')

    def delete(self,condition):
        '''
        delete samples.
        
        Args:
            condition: A dict determines which samples should be delete.
        Return:
            None
        '''
        conforming_idx = self._deal_condition(condition)
        keep_idx = list(set([i for i in range(self.len_data)]) - set(conforming_idx))
        self.dataset = self.dataset[keep_idx]
        for k in range(self.info.keys()):
            self.info[k] = self.info[k][keep_idx]

    # get information or values
    def _get_data(self,idx_list):
        '''
        get data from dataset
        Args:
            idx_list: required dataset index
        Return:
            A list contained required data
        '''
        if self.save_in_piece:
            data_path = self.save_path + 'DataSet_' + self.name + '/'
            r_data = []
            for i in idx_list:
                r_data.append(np.load(data_path + self.info['file_name'][i] + '.npy'))
            return r_data
        else:
            return self.dataset[idx_list]

    def get_value_attribute(self,attribute):
        '''
        get values under the given attribute of each data
        Args:
            attribute: A str mapping the attribute of dataset.
                Return error and all attribute of dataset if the given attribute does
                not exist.
        Return:
            A list of values under the given attribute with the same order as samples in dataset.
        '''
        try:
            return self.info[attribute]
        except KeyError:
            raise ValueError('The given attribute does not exist in index, and the attributes of this dataset \
                is '+ str(list(self.info.keys())))

    def get_value(self,attribute,condition={}):
        '''
        get corresponding values.
        
        Args:
            attribute: A string describes the values returned.
            condition: A dict determines the values of which samples should be returned.
        Return:
            A list contrained values by given attribute and condition.
        '''
        conforming_idx = self._deal_condition(condition)
        if attribute == 'data':
            return self._get_data(conforming_idx)
        else:
            try:
                return self.info[attribute][conforming_idx]
            except KeyError:
                raise ValueError('The given attribute does not exist in index, and the attributes of this dataset \
                    is '+ str(list(self.info.keys())))

    def get_dataset(self,condition={}):
        '''
        get corresponding dataset.
        
        Args:
            condition: A dict determines the values of which samples should be returned.
        Return:
            A DataSet contrained values by given condition.
        '''
        conforming_idx = self._deal_condition(condition)
        temp_data = self._get_data(conforming_idx)
        temp_info = OrderedDict()
        for k in self.info.keys():
            temp_info[k] = self.info[k][conforming_idx]
        temp_len_data = len(conforming_idx)
        return DataSet(name='temp',info=temp_info,dataset=temp_data,len_data=temp_len_data,load_name=self.load_name)

    def get_random_choice(self):
        '''
        get a random sample.
        
        Args:
            None
        Return:
            A dict like {Attribute_1:Values,...}.
        '''
        r = OrderedDict()
        ran_idx = random.randint(0,self.len_data-1)
        r['data'] = self._get_data(ran_idx)
        for k in self.info.keys():
            r[k] = self.info[k][ran_idx]
        return r

    def get_random_samples(self,n=1):
        '''
        get a random DataSet.
        
        Args:
            None
        Return:
            A Dataset with same index but only one sample.
        '''
        ran_idx = [random.randint(0,self.len_data-1) for _ in range(n)]
        temp_data = self._get_data(ran_idx)
        temp_info = OrderedDict()
        for k in self.info.keys():
            temp_info[k] = self.info[k][ran_idx]
        temp_len_data = len(ran_idx)
        return DataSet(name='temp',info=temp_info,dataset=temp_data,len_data=temp_len_data,load_name=self.load_name)

    def save(self,piece=False):
        '''
        Save this DataSet as .pkl file.
        
        Args:
            piece: A bool to decide whether save dataset as piece.
        Return:
            None
        '''
        assert self.name != ''
        assert self.save_path != ''
        if piece:
            self.save_in_piece = True
            if self.name != self.load_name:
                data_path = self.save_path + 'DataSet_' + self.name
                if os.path.exists(data_path):
                    shutil.rmtree(data_path)
                    os.makedirs(data_path)
                else:
                    os.makedirs(data_path)
                data_path += '/'
                for i,x in enumerate(self.dataset):
                    if isinstance(x,np.ndarray):
                        np.save(data_path + self.info['file_name'][i] + '.npy', x)
                    elif x == None:
                        shutil.copyfile(
                            self.save_path+self.load_name+'/'+self.info['file_name'][i]+'.npy',
                            data_path + self.info['file_name'][i]+'.npy'
                            )
                    else:
                        raise ValueError
            else:
                data_path = self.save_path + 'DataSet_' + self.name
                origin_files = os.listdir(self.save_path + self.load_name + '/')
                origin_files = [x[:-4] for x in origin_files if '.npy' in x]
                add_list = list(set(self.info['file_name']) - set(origin_files))
                del_list = list(set(origin_files) - set(self.info['file_name']))
                for x in add_list:
                    save_data = self.dataset[self.info['file_name'].index(x)]
                    assert save_data != None
                    np.save(data_path + x + '.npy', save_data)
                for x in del_list:
                    os.remove(self.save_path + self.load_name + '/' + x + '.npy')
        else:
            pickle.dump(self.dataset, open(self.save_path + 'DataSet_' +
                                        self.name + '.pkl', 'wb'), True)
        pd.DataFrame(self.info).to_csv(self.save_path + 'DataSet_' + self.name + 'info.csv')
        print('dataset ', self.name, ' has benn saved\n')

    def load(self,name=''):
        '''
        Load this DataSet with name and path known, which should be given when initialize DataSet class.
        
        Args:
            name: The name of DataSet.
        Return:
            None
        '''
        if name != '':
            self.name = name
        assert self.name != ''
        assert self.save_path != ''
        full_name = self.save_path + 'DataSet_' + self.name + '.pkl'
        if os.path.exists(full_name):
            self.dataset = pickle.load(open(full_name, 'rb'))
            self.info = OrderedDict(pd.read_csv(self.save_path + 'DataSet_' + self.name + 'info.csv'))
            for k in self.info.keys():
                self.info[k] = list(self.info[k])
            self.len_data = self.info.pop(list(self.info.keys())[0])[-1] + 1
            self.save_in_piece = False
        elif os.path.exists(full_name[:-4]):
            self.info = OrderedDict(pd.read_csv(self.save_path + 'DataSet_' + self.name + 'info.csv'))
            for k in self.info.keys():
                self.info[k] = list(self.info[k])
            self.len_data = self.info.pop(list(self.info.keys())[0])[-1] + 1
            self.save_in_piece = True
            self.dataset = [None]*self.len_data
            self.load_name = name
        else:
            raise ValueError('required dataset does not exist!')
        print('dataset ', self.name, ' has been load')

    @staticmethod
    def load_dataset(name):
        '''
        Load this DataSet with name and default path './data/'.
        
        Args:
            name: The name of DataSet.
        Return:
            DataSet
        '''
        r_dataset = DataSet()
        r_dataset.load(name)
        return r_dataset

def make_phm_dataset():
    RUL_dict = {'Bearing1_1':0,'Bearing1_2':0,
                'Bearing2_1':0,'Bearing2_2':0,
                'Bearing3_1':0,'Bearing3_2':0,
                'Bearing1_3':573,'Bearing1_4':33.9,'Bearing1_5':161,'Bearing1_6':146,'Bearing1_7':757,
                'Bearing2_3':753,'Bearing2_4':139,'Bearing2_5':309,'Bearing2_6':129,'Bearing2_7':58,
                'Bearing3_3':82}
    phm_dataset = DataSet(name='phm_data',
                        index=['bearing_name','RUL','quantity','data'])
    source_path = './PHM/'
    for path_1 in ['Learning_set/','Test_set/']:
        bearings_names = os.listdir(source_path + path_1)
        bearings_names.sort()
        for bearings_name in bearings_names:
            file_names = os.listdir(source_path + path_1 + bearings_name + '/')
            file_names.sort()
            bearing_data = np.array([])
            for file_name in file_names:
                if 'acc' in file_name:
                    df = pd.read_csv(source_path + path_1 + bearings_name + '/'\
                                    + file_name,header=None)
                    data = np.array(df.loc[:,4:6])
                    data = data[np.newaxis,:,:]
                    if bearing_data.size == 0:
                        bearing_data = data
                    else:
                        bearing_data = np.append(bearing_data,data,axis=0)
        
            phm_dataset.append([bearings_name,RUL_dict[bearings_name],bearing_data.shape[0],bearing_data])
            print(bearings_name,'has been appended.')

    phm_dataset.save()

def make_paderborn_dataset():
    info = OrderedDict()
    info['file_name'] = []
    info['load'] = []
    info['speed'] = []
    info['fault_place'] = []
    info['fault_cause'] = []
    info['state'] = []
    info['No'] = []
    paderborn_dataset = DataSet(
        name='paderborn_data',
        info=info
    )
    source_path = 'E:/cyh/data_sum/temp/德data/dataset/'
    artificial_fault = ['KI01','KI03','KI05','KI07','KI08',
                'KA01','KA03','KA05','KA06','KA07']
    state = {
        'K001':'>50',
        'K002':'19',
        'K003':'1',
        'K004':'5',
        'K005':'10',
        'K006':'16',
        'KA01':'EMD',
        'KA03':'Electric Engraver',
        'KA05':'Electric Engraver',
        'KA06':'Electric Engraver',
        'KA07':'Drilling',
        'KA08':'Drilling',
        'KA09':'Drilling',
        'KA04':'Pitting',
        'KA15':'Plastic Deform',
        'KA16':'Pitting',
        'KA22':'Pitting',
        'KA30':'Plastic Deform',
        'KI01':'EMD',
        'KI03':'Electric Engraver',
        'KI05':'Electric Engraver',
        'KI07':'Electric Engraver',
        'KI08':'Electric Engraver',
        'KI04':['Pitting','Plastic Deform'],
        'KI14':['Pitting','Plastic Deform'],
        'KI16':'Pitting',
        'KI17':'Pitting',
        'KI18':'Pitting',
        'KI21':'Pitting',
        'KB23':'Pitting',
        'KB24':'Pitting',
        'KB27':'Plastic Deform'
    }
    file_names = os.listdir(source_path)
    file_names.sort()
    for file_name in file_names:
        temp_data = sio.loadmat(source_path + file_name)
        temp_data = temp_data[file_name.replace('.mat','')]
        temp_data = temp_data['Y']
        temp_data = temp_data[0][0][0][6][2][0]
        temp_fault_cause = 'artificial' if file_name[12:16] in artificial_fault else 'real'
        temp_append_sample = [
            temp_data,
            file_name.replace('.mat',''),
            file_name[4:11],
            file_name[0:3],
            file_name[12:16],
            temp_fault_cause,
            state[file_name[12:16]],
            file_name[17:].replace('.mat','')
        ]
        paderborn_dataset.append(temp_append_sample)
        print(file_name,'has been appended.')

    paderborn_dataset.save(piece=True)

def make_ims_dataset():
    fault_bearing = {'1st_test':OrderedDict({4:'3_x',5:'3_y',6:'4_x',7:'4_y'}), '2nd_test':[0], '4th_test':[2]}
    ims_dataset = DataSet(name='ims_data', index=['set_No','bearing_No','record_time','data'])
    source_path = 'E:/cyh/data_sum/temp/IMS data/'

    for dir_name in fault_bearing.keys():
        # if isinstance(fault_bearing[dir_name],dict):
        #     all_samples = []
        #     for k in fault_bearing[dir_name].keys():
        #         all_samples.append([dir_name, fault_bearing[dir_name][k], [], []])
        if isinstance(fault_bearing[dir_name], dict):
            channels = list(fault_bearing[dir_name].keys())
        elif isinstance(fault_bearing[dir_name], list):
            channels = fault_bearing[dir_name]
        record_time = []
        record_data = np.array([])

        names = os.listdir(source_path + dir_name + '/')
        names.sort()
        for name in names:
            print(name)
            record_time.append(name.replace('.txt',''))
            temp_data = np.loadtxt(source_path + dir_name + '/' + name)
            if record_data.size == 0:
                record_data = temp_data[:,channels][np.newaxis,:,:]
            else:
                record_data = np.append(record_data, temp_data[:,channels][np.newaxis,:,:], axis=0)
        
        if isinstance(fault_bearing[dir_name], dict):
            append_samples = []
            for i,k in enumerate(fault_bearing[dir_name].keys()):
                append_samples.append([dir_name, fault_bearing[dir_name][k], record_time, record_data[:,:,i]])
        elif isinstance(fault_bearing[dir_name], list):
            append_samples = []
            for i,x in enumerate(fault_bearing[dir_name]):
                append_samples.append([dir_name, str(x), record_time, record_data[:,:,i]])

        for sample in append_samples:
            ims_dataset.append(sample)

    ims_dataset.save()


if __name__ == '__main__':
    # make_phm_dataset()
    # dataset = DataSet.load_dataset('phm_data')
    # dataset._save_info()
    make_paderborn_dataset()
    # make_ims_dataset()
    # dataset = DataSet.load_dataset('ims_data')
    print('1')
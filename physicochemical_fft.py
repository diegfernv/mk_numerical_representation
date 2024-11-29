import pandas as pd 
from scipy.fft import fft
import numpy as np
import math, os, argparse

class PhysicochemicalEncoder:

    def __init__(self,
                 dataset=None,
                 sep_dataset=",",
                 property_encoder="Group_0",
                 dataset_encoder=None,
                 name_column_seq="sequence",
                 columns_to_ignore=[]):

        self.dataset = dataset
        self.sep_dataset = sep_dataset

        self.property_encoder = property_encoder
        self.dataset_encoder = dataset_encoder
        self.name_column_seq = name_column_seq
        self.columns_to_ignore = columns_to_ignore

        self.possible_residues = [
            'A',
            'C',
            'D',
            'E',
            'F',
            'G',
            'H',
            'I',
            'N',
            'K',
            'L',
            'M',
            'P',
            'Q',
            'R',
            'S',
            'T',
            'V',
            'W',
            'Y'
        ]

        self.df_data_encoded = None

        self.status = False
        self.message= ""

    def run_process(self):
        self.__make_validations()

        if self.status == True:
            self.zero_padding = self.__check_max_size()
            self.__encoding_dataset()
            self.message = "ENCODING OK"
        
    def __check_columns_in_df(
            self,
            check_columns=None,
            columns_in_df=None):

        response_check = True

        for colum in check_columns:
            if colum not in columns_in_df:
                response_check=False
                break
        
        return response_check
    
    def __make_validations(self):

        # read the dataset with encoders
        self.dataset_encoder.index = self.dataset_encoder['residue']
        
        # check input dataset
        if self.name_column_seq in self.dataset.columns:
            
            if isinstance(self.columns_to_ignore, list):

                if len(self.columns_to_ignore)>0:
                    
                    response_check = self.__check_columns_in_df(
                        columns_in_df=self.dataset.columns.values,
                        check_columns=self.columns_to_ignore
                    )
                    if response_check == True:
                        self.status=True
                    else:
                        self.message = "ERROR: IGNORE COLUMNS NOT IN DATASET COLUMNS"   
                else:
                    pass
            else:
                self.message = "ERROR: THE ATTRIBUTE columns_to_ignore MUST BE A LIST"
        else:
            self.message = "ERROR: COLUMN TO USE AS SEQUENCE IS NOT IN DATASET COLUMNS"    

    def __check_residues(self, residue):
        if residue in self.possible_residues:
            return True
        else:
            return False

    def __encoding_residue(self, residue):

        if self.__check_residues(residue):
            return self.dataset_encoder[self.property_encoder][residue]
        else:
            return False

    def __check_max_size(self):
        size_list = [len(seq) for seq in self.dataset[self.name_column_seq]]
        return max(size_list)

    def __encoding_sequence(self, sequence):

        sequence = sequence.upper()
        sequence_encoding = []

        for i in range(len(sequence)):
            residue = sequence[i]
            response_encoding = self.__encoding_residue(residue)
            if response_encoding != False:
                sequence_encoding.append(response_encoding)

        # complete zero padding
        for k in range(len(sequence_encoding), self.zero_padding):
            sequence_encoding.append(0)

        return sequence_encoding

    def __encoding_dataset(self):

        #print("Start encoding process")
        if len(self.columns_to_ignore)>0:
            df_columns_ignore = self.dataset[self.columns_to_ignore]
            dataset_to_encode = self.dataset.drop(columns=self.columns_to_ignore)
        else:
            df_columns_ignore=None
            dataset_to_encode = self.dataset

        print("Encoding and Processing results")

        matrix_data = []
        for index in dataset_to_encode.index:
            sequence_encoder = self.__encoding_sequence(sequence=dataset_to_encode[self.name_column_seq][index])
            matrix_data.append(sequence_encoder)

        print("Creating dataset")
        header = ['p_{}'.format(i) for i in range(len(matrix_data[0]))]
        print("Export dataset")

        self.df_data_encoded = pd.DataFrame(matrix_data, columns=header)

        if len(self.columns_to_ignore)>0:
            self.df_data_encoded = pd.concat([self.df_data_encoded, df_columns_ignore], axis=1)

class FFTTransform:

    def __init__(
            self,
            dataset=None,
            size_data=None,
            columns_to_ignore=[]):
        
        self.size_data = size_data
        self.dataset = dataset
        self.columns_to_ignore = columns_to_ignore

        self.init_process()

    def __processing_data_to_fft(self):

        print("Removing columns data")
        
        if len(self.columns_to_ignore) >0:
            self.data_ignored = self.dataset[self.columns_to_ignore]
            self.dataset = self.dataset.drop(columns=self.columns_to_ignore)
    
    def __get_near_pow(self):

        print("Get near pow 2 value")
        list_data = [math.pow(2, i) for i in range(1, 20)]
        stop_value = list_data[0]

        for value in list_data:
            if value >= self.size_data:
                stop_value = value
                break

        self.stop_value = int(stop_value)
    
    def __complete_zero_padding(self):

        print("Apply zero padding")
        list_df = [self.dataset]
        for i in range(self.size_data, self.stop_value):
            column = [0 for k in range(len(self.dataset))]
            key_name = "p_{}".format(i)
            df_tmp = pd.DataFrame()
            df_tmp[key_name] = column
            list_df.append(df_tmp)

        self.dataset = pd.concat(list_df, axis=1)
    

    def init_process(self):
        self.__processing_data_to_fft()
        self.__get_near_pow()
        self.__complete_zero_padding()

    def __create_row(self, index):
        row =  self.dataset.iloc[index].tolist()
        return row
    
    def __apply_FFT(self, index):

        row = self.__create_row(index)
        T = 1.0 / float(self.stop_value)
        yf = fft(row)

        xf = np.linspace(0.0, 1.0 / (2.0 * T), self.stop_value // 2)
        yf = np.abs(yf[0:self.stop_value // 2])
        return [value for value in yf]


    def encoding_dataset(self):

        matrix_response = []
        for index in self.dataset.index:
            row_fft = self.__apply_FFT(index)
            matrix_response.append(row_fft)

        print("Creating dataset")
        header = ['p_{}'.format(i) for i in range(len(matrix_response[0]))]
        print("Export dataset")
        df_fft = pd.DataFrame(matrix_response, columns=header)
        
        if len(self.columns_to_ignore)>0:

            df_fft = pd.concat([df_fft, self.data_ignored], axis=1)

        return df_fft

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file", required=True)
    parser.add_argument("-r", "--response", help="Response column", required=True)
    parser.add_argument("-s", "--sequence", help="Sequence column", required=True)
    parser.add_argument("-o", "--output", help="Output path", required=True)
    parser.add_argument("-p", "--property", help="Property name", default="ANDN920101")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.rename(columns={args.sequence: "sequence", args.response: "response"}, inplace=True)

    aaindex = pd.read_csv("aaindex_encoders.csv")
    aaindex.index = aaindex['residue']

    physicochemical_instance = PhysicochemicalEncoder(
        dataset=df,
        sep_dataset=",",
        property_encoder=args.property,
        dataset_encoder=aaindex,
        name_column_seq="sequence",
        columns_to_ignore=["response"]
    )
    physicochemical_instance.run_process()

    physicochemical_instance.df_data_encoded.to_csv("{}/physicochemical_{}.csv".format(args.output, args.property), index=False)

    fft_instance = FFTTransform(
        dataset=physicochemical_instance.df_data_encoded,
        size_data=len(physicochemical_instance.df_data_encoded.columns)-1,
        columns_to_ignore=["response"]
    )

    df_fft = fft_instance.encoding_dataset()

    df_fft.to_csv("{}/fft_{}.csv".format(args.output, args.property), index=False)


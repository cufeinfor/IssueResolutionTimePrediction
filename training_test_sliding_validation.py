from read_feature_set import *

def getData_all(fPath):

    if os.path.exists(fPath):
        data = pd.read_csv(fPath)
        new_col1 = FEATURE_NAME_LIST.copy()
        new_col1.append('label')
        data = np.array(data[new_col1])
        return data
    else:
        print('No such file or directory!')


# training-test sliding validation method
def getData_slidingValidation_list(dataset, train_step=20, test_step=5):
    # Read  data
    train_dataDict = getData_slidingValidation(dataset, train_step, test_step)
    setNums = len(train_dataDict.keys()) / 4
    train_data_x_list = []
    train_data_y_list = []
    train_validate_data_x_list = []
    train_validate_data_y_list = []
    for i in range(1, int(setNums + 1)):
        trainMatrix = train_dataDict[str(i) + 'train']
        trainClass = train_dataDict[str(i) + 'trainclass']
        testMatrix = train_dataDict[str(i) + 'test']
        testClass = train_dataDict[str(i) + 'testclass']
        train_data_x_list.append(trainMatrix)
        train_data_y_list.append(trainClass)
        train_validate_data_x_list.append(testMatrix)
        train_validate_data_y_list.append(testClass)
    return train_data_x_list, train_data_y_list, train_validate_data_x_list, train_validate_data_y_list

# 划分验证数据集，training-test sliding validation method
def getData_slidingValidation(dataset, train_step=36,test_step=4):
    # Reading data
    dataMatrix = np.array(dataset)
    # Get the features of each sample and the class label
    rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
    sampleData = []
    sampleClass = []
    for i in range(0, rowNum):
        tempList = list(dataMatrix[i, :])
        sampleClass.append(tempList[-1])
        sampleData.append(tempList[:-1])
    sampleM = np.array(sampleData)
    sampleFeatureNum = len(sampleM[1]) #the number of features
    classM = np.array(sampleClass)
    new_sampleM = split_data(sampleM,100)
    new_classM = split_data(classM,100)
    setDict = {}
    count = 1
    for i in range(0,int((100-train_step-test_step)/test_step)+1):
        front_index = test_step*i+train_step
        behind_index = front_index + test_step
        print(i*test_step,front_index)
        print(front_index,behind_index)
        trainSTemp = new_sampleM[i*test_step:front_index]
        trainCTemp = new_classM[i*test_step:front_index]
        testSTemp = new_sampleM[front_index:behind_index]
        testCTemp = new_classM[front_index:behind_index]
        # Generate training sets
        trainSTemp = trainSTemp.reshape(-1, sampleFeatureNum)
        setDict[str(count) + 'train'] = trainSTemp
        trainCTemp = trainCTemp.reshape(-1)
        setDict[str(count) + 'trainclass'] = trainCTemp
        # Generate testing sets
        testSTemp = testSTemp.reshape(-1, sampleFeatureNum)
        setDict[str(count) + 'test'] = testSTemp
        testCTemp = testCTemp.reshape(-1)
        setDict[str(count) + 'testclass'] = testCTemp
        count += 1
    return setDict

def split_data(data,k_subset):
    index_subset = int(len(data)/k_subset)
    new_data = []
    for i in range(0,k_subset):
        front_index = (i+0)*index_subset
        behind_index = (i+1)*index_subset
        subset = data[front_index:behind_index]
        new_data.append(subset)
    return np.array(new_data)
from . import swigknowhere
from .swigknowhere import Status
from .swigknowhere import GetBinarySet, GetNullDataSet, GetNullBitSetView
# from .swigknowhere import BruteForceSearch, BruteForceRangeSearch
import numpy as np


def CreateIndex(name, version, type):
    if type == np.float32:
        return swigknowhere.IndexWrapFloat(name, version)
    if type == np.float16:
        return swigknowhere.IndexWrapFP16(name, version)


def GetCurrentVersion():
    return swigknowhere.CurrentVersion()


def CreateBitSet(bits_num):
    return swigknowhere.BitSet(bits_num)


def Load(binset, file_name):
    return swigknowhere.Load(binset, file_name)


def Dump(binset, file_name):
    return swigknowhere.Dump(binset, file_name)


def ArrayToDataSet(arr):
    if arr.ndim == 1:
        return swigknowhere.Array2DataSetIds(arr)
    if arr.ndim == 2:
        if arr.dtype == np.int32:
            return swigknowhere.Array2DataSetI(arr)
        if arr.dtype == np.float32:
            return swigknowhere.Array2DataSetF(arr)
        if arr.dtype == np.float16:
            # 转float16 to float32
            arr = arr.astype(np.float32)
            return swigknowhere.Array2DataSetFP16(arr)
    raise ValueError(
        """
        ArrayToDataSet only support numpy array dtype float32 and int32 and float16.
        """
    )


def DataSetToArray(ans):
    dim = swigknowhere.DataSet_Dim(ans)
    rows = swigknowhere.DataSet_Rows(ans)
    dis = np.zeros([rows, dim]).astype(np.float32)
    ids = np.zeros([rows, dim]).astype(np.int32)
    swigknowhere.DataSet2Array(ans, dis, ids)
    return dis, ids


def RangeSearchDataSetToArray(ans):
    rows = swigknowhere.DataSet_Rows(ans)
    lims = np.zeros(
        [
            rows + 1,
        ],
        dtype=np.int32,
    )
    swigknowhere.DumpRangeResultLimits(ans, lims)
    dis = np.zeros(
        [
            lims[-1],
        ],
        dtype=np.float32,
    )
    swigknowhere.DumpRangeResultDis(ans, dis)
    ids = np.zeros(
        [
            lims[-1],
        ],
        dtype=np.int32,
    )
    swigknowhere.DumpRangeResultIds(ans, ids)

    dis_list = []
    ids_list = []
    for idx in range(rows):
        dis_list.append(dis[lims[idx] : lims[idx + 1]])
        ids_list.append(ids[lims[idx] : lims[idx + 1]])

    return dis_list, ids_list


def GetVectorDataSetToArray(ans):
    dim = swigknowhere.DataSet_Dim(ans)
    rows = swigknowhere.DataSet_Rows(ans)
    data = np.zeros([rows, dim]).astype(np.float32)
    swigknowhere.DataSetTensor2Array(ans, data)
    return data


def GetBinaryVectorDataSetToArray(ans):
    dim = int(swigknowhere.DataSet_Dim(ans) / 32)
    rows = swigknowhere.DataSet_Rows(ans)
    data = np.zeros([rows, dim]).astype(np.int32)
    swigknowhere.BinaryDataSetTensor2Array(ans, data)
    return data

def SetSimdType(type):
    swigknowhere.SetSimdType(type)

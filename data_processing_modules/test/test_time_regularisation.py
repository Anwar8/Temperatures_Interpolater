import pytest
import numpy as np

from ..time_regularisation import *


def test_linearly_interpolate_correctness_positive():
    """
    Checks that the function `linearly_interpolate` interpolates
    a positive value correctly for a line with positive slope.
    """
    x_i, y_i = 0.0, 0.0
    x_f, y_f = 2.0, 2.0
    x, y = 1.0, 1.0
    assert np.isclose(linearly_interpolate(x, x_i, x_f, y_i, y_f), y, rtol=1e-4)

def test_linearly_interpolate_correctness_negative():
    """
    Checks that the function `linearly_interpolate` interpolates
    a negative value correctly for a line with negative slope.
    """
    x_i, y_i = 0.0, 0.0
    x_f, y_f = 2.0, -2.0
    x, y = 1.0, -1.0
    assert np.isclose(linearly_interpolate(x, x_i, x_f, y_i, y_f), y, rtol=1e-4)

def test_linearly_interpolate_x_between_points():
    """
    Ensures that a value error is raised if x was not within x_i and x_f.
    """
    x_i, y_i = -1.0, -1.0
    x_f, y_f = 2.0, 2.0
    x = 3.0
    with pytest.raises(ValueError):
        _ = linearly_interpolate(x, x_i, x_f, y_i, y_f)
    
def test_linearly_interpolate_wrong_boundaries():
    """
    Ensures that a value error is raised if x_i was not smaller than x_f.
    """
    x_i, y_i = 10.0, -1.0
    x_f, y_f = 2.0, 2.0
    x = 3.0
    with pytest.raises(ValueError):
        _ = linearly_interpolate(x, x_i, x_f, y_i, y_f)
    
def test_regularise_data_regular_x_type():
    """
    Ensures that a type error is raised if regular x was not a numpy ndarray.
    """
    regular_x = [8.0, 3.4, 2.6]
    list_of_np_arrays = [np.zeros((5,2)), np.zeros((5,2))]
    with pytest.raises(TypeError):
        _ = regularise_data(regular_x, list_of_np_arrays) 

def test_regularise_data_list_of_ndarrays_list_type():
    """
    Ensures that a type error is raised if the second argument for `regularise_data`
    was not a python list.
    """
    regular_x = np.array([8.0, 3.4, 2.6])
    list_of_np_arrays = np.zeros((5,2))
    with pytest.raises(TypeError):
        _ = regularise_data(regular_x, list_of_np_arrays) 

def test_regularise_data_list_of_ndarrays_ndarray_type():
    """
    Ensures that a type error is raised if the second argument for `regularise_data`
    was not a python list of numpy ndarrays.
    """
    regular_x = np.array([8.0, 3.4, 2.6])
    list_of_np_arrays = [6.0, 8.0, -100.0]
    with pytest.raises(TypeError):
        _ = regularise_data(regular_x, list_of_np_arrays) 


def test_regularise_data_input_size():
    """
    Ensures that a ValueError is raised if the various numpy arrays 
    in the second input do not have at least two columns.
    """
    regular_x = np.array([8.0, 3.4, 2.6])
    list_of_np_arrays = [np.zeros((5,2)), np.zeros((8,1))]
    with pytest.raises(ValueError):
        _ = regularise_data(regular_x, list_of_np_arrays) 

def test_regularise_data_output_size():
    """
    Asserts that the output size is as expected: the same number of
    rows as regular_x, and the same number of columns as the number
    of np.ndarrays in arraylist and an extra column corresponding to
    the column containing the regular x.
    """
    regular_x = np.array([0.0, 1.5, 3.0])
    list_of_np_arrays = [np.zeros((5,2)), np.zeros((8,2)), np.zeros((10,2))]
    assert regularise_data(regular_x, list_of_np_arrays).shape == (3, 4)

def test_regularise_data_output_time():
    """
    Ensures that the first column of the output is exactly the given
    regular_x array.
    """
    regular_x = np.array([0.0, 1.5, 3.0])
    list_of_np_arrays = [np.zeros((5,2)), np.zeros((8,2)), np.zeros((10,2))]
    assert np.allclose(regularise_data(regular_x, list_of_np_arrays)[:,0], 
        regular_x, rtol=1e-4)

def test_regularise_data_output_correctness_ascending():
    """
    Ensures that the output from `regularise_data_output` is correct
    when gien a linearly ascending input. Both more and fewer points
    are given in the data to be regularised than the regular_x points.
    """
    regular_x = np.array([0.0, 0.5, 1.0, 1.5])
    first_time_series = np.array([[0.0, 0.25, 0.75, 1.25, 1.5], [0.0, 0.25, 0.75, 1.25, 1.5]]).T
    second_time_series = np.array([[0.0, 1.5], [1.0, 16.0]]).T
    list_of_np_arrays = [first_time_series, second_time_series]
    correct_output = np.array([[0.0, 0.5, 1.0, 1.5], 
                                  [0.0, 0.5, 1.0, 1.5],
                                  [1.0, 6.0, 11.0, 16.0]]).T
    print("regularised x =\n{}".format(regularise_data(regular_x, list_of_np_arrays)))
    assert np.allclose(regularise_data(regular_x, list_of_np_arrays), 
        correct_output, rtol=1e-4)

def test_interpolate_between_columns_columns_type():
    """
    Ensures that a TypeError is raised if wrong columns type is passed
    """
    columns = [8.0, 3.4, 2.6]
    column_titles_array = np.ones((1,3))
    inteprolation_array = np.ones((1,4))
    with pytest.raises(TypeError):
        _ = interpolate_between_columns(column_titles_array, columns, inteprolation_array)

def test_interpolate_between_columns_column_titles_type():
    """
    Ensures that a TypeError is raised if wrong column_titles_array type is passed
    """
    columns = np.ones((10,3))
    column_titles_array = [5.0, 2.0, -1.7]
    inteprolation_array = np.ones((1,4))
    with pytest.raises(TypeError):
        _ = interpolate_between_columns(column_titles_array, columns, inteprolation_array)

def test_interpolate_between_columns_interpolation_array_type():
    """
    Ensures that a TypeError is raised if wrong interpolation_array type is passed
    """
    columns = np.ones((10,3))
    column_titles_array = np.ones((1,3))
    inteprolation_array = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(TypeError):
        _ = interpolate_between_columns(column_titles_array, columns, inteprolation_array)

def test_interpolate_between_columns_columns_and_titles_size():
    """
    Ensures that a ValueError is raised if there are not exactly the same
    number of columns and column_titles
    """
    columns = np.ones((10,3))
    column_titles_array = np.ones((1,4))
    inteprolation_array = np.ones((1,3))
    with pytest.raises(ValueError):
        _ = interpolate_between_columns(column_titles_array, columns, inteprolation_array)

def test_interpolate_between_columns_column_titles_size():
    """
    Ensures that a ValueError is raised if there are not exactly one row
    of column titles
    """
    columns = np.ones((10,3))
    column_titles_array = np.ones((2,3))
    inteprolation_array = np.ones((1,3))
    with pytest.raises(ValueError):
        _ = interpolate_between_columns(column_titles_array, columns, inteprolation_array)
        
def test_interpolate_between_columns_interpolation_array_size():
    """
    Ensures that a ValueError is raised if there are not exactly one row
    of interpolation array
    """
    columns = np.ones((10,3))
    column_titles_array = np.ones((1,3))
    inteprolation_array = np.ones((2,3))
    with pytest.raises(ValueError):
        _ = interpolate_between_columns(column_titles_array, columns, inteprolation_array)

def test_interpolate_between_columns_column_titles_order():
    """
    Ensures that a ValueError is raised if the ordered column titles are
    not ordered in ascending order
    """
    columns = np.ones((10,3))
    column_titles_array = np.array([5.0, 2.0, 1.0])
    inteprolation_array = np.ones((1,3))
    with pytest.raises(ValueError):
        _ = interpolate_between_columns(column_titles_array, columns, inteprolation_array)

def test_interpolate_between_columns_first_column_titles_interpolation_values():
    """
    Ensures that a ValueError is raised if the first requested interpolation
    is smaller than the first given data point.
    """
    columns = np.ones((10,3))
    column_titles_array = np.array([5.0, 10.0, 15.0])
    inteprolation_array = np.array([1.0, 5.0, 10.0])
    with pytest.raises(ValueError):
        _ = interpolate_between_columns(column_titles_array, columns, inteprolation_array)

def test_interpolate_between_columns_last_column_titles_interpolation_values():
    """
    Ensures that a ValueError is raised if the last requested interpolation
    is larger than the last given data point.
    """
    columns = np.ones((10,3))
    column_titles_array = np.array([5.0, 10.0, 15.0])
    inteprolation_array = np.array([5.0, 15.0, 25.0])
    with pytest.raises(ValueError):
        _ = interpolate_between_columns(column_titles_array, columns, inteprolation_array)

def test_interpolate_between_columns_output_shape():
    """
    Ensures that the output will have the same number of rows as the columns
    array, but the same number of columns as the interpolation_arrray
    """
    columns = np.ones((10,3))
    column_titles_array = np.array([0.0, 2.0, 4.0])
    inteprolation_array = np.ones((1,4))
    output_shape = interpolate_between_columns(column_titles_array, columns, inteprolation_array).shape
    assert  output_shape == (10,4)

def test_interpolate_between_columns_output_correctness():
    """
    Ensures that the output is correct for an interpolation problem with
    a known correct solution.
    """
    columns = np.array([[10.0, 20.0, 0.0],[20.0, 40.0, -40.0],[30.0, 60.0, -120.0]])
    correct = np.array([[15.0, 10.0], [30.0, 0.0], [45.0, -30.0]])
    column_titles_array = np.array([0.0, 2.0, 4.0])
    inteprolation_array = np.array([1.0, 3.0])
    output = interpolate_between_columns(column_titles_array, columns, inteprolation_array)
    assert  np.allclose(output, correct, rtol=1e-4)

def test_interpolate_between_columns_input_equal_output_correctness():
    """
    Ensures that the output is the same as the input columns when the
    interpolation array is the same as the orignal description.
    """
    columns = np.array([[10.0, 20.0, 0.0],[20.0, 40.0, -40.0],[30.0, 60.0, -120.0]])
    correct = np.array([[10.0, 20.0, 0.0], [20.0, 40.0, -40.0], [30.0, 60.0, -120.0]])
    column_titles_array = np.array([0.0, 2.0, 4.0])
    inteprolation_array = np.array([0.0, 2.0, 4.0])
    output = interpolate_between_columns(column_titles_array, columns, inteprolation_array)
    assert  np.allclose(output, correct, rtol=1e-4)
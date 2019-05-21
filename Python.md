# PYTHON FOR MACHINE LEARNING


![1_7EUX9QIjq2x1JyFKcjhXsA](https://user-images.githubusercontent.com/17926361/58113851-1aa93e00-7c14-11e9-969c-1234946264c0.png)

#### Python Data Types
In Python, common data types are float (floating point), int (integer), str (string), bool (Boolean), list, and dict (dictionary).

* float - used for real numbers.
* int - used for integers.
* str - used for texts. 
       We can define strings using single quotes'value', double quotes"value", or triple quotes"""value""". 
       The triple quoted strings can be on multiple lines, the new lines will be included in the value of the variable. 
       They’re also used for writing function documentation.
* bool - used for truthy values. Useful to perform a filtering operation on a data.
* list - used to store a collection of values.
* dict - used to store a key-values pairs.

Here we can look at some examples with creating a floating points, intergers, strings and booleans in Python.```

 ![Screenshot (265)](https://user-images.githubusercontent.com/17926361/58115054-d8353080-7c16-11e9-9a6b-d3f6e3a08295.png)
  
__Python Lists__
* Python list is a basic sequence type.
* One list can contain values of any type. 
* It is possible that one list contains another nested lists for its values. 
* It’s not commonly used, but you can have a list with a mix of Python types. You can create a new one using square brackets like this:

`fruits = ["pineapple", "apple", "lemon", "strawberry", "orange", "kiwi"]`

__Subsetting Lists__
* Indexes are used to get element or elements from the list.
* In Python, the indexes start from 0. Therefore, the first element in the list will have an index 0.
* Negative indexes are also used to access elements.The last element in the list will have an index -1,
  the one before the last one will have an index -2 and so on. 
* _list slicing_ which can be used to get multiple elements from a list.
  We can use it like this: 
      `sliceable[start_index:end_index:step].`
      
      
  * The __start_index__ is the beginning index of the slice, the element at this index will be included to the result,
        the default value is 0.
  * The __end_index__ is the end index of the slice, the element at this index will not be included to the result, the default
        value will be the length of the list. Also, the default value can be - length of the list -1 if the step is negative.
        If you skip this, you will get all the elements from the start index to the end.
  * The step is the amount by which the index increases, 
        the default value is 1. If we set a negative value for the step, we’ll move backward.
        
![Screenshot (267)](https://user-images.githubusercontent.com/17926361/58116240-74603700-7c19-11e9-83c4-74f4e3d7f10a.png)

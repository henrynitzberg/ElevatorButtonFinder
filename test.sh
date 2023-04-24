#!/bin/bash

echo "Test Num:"
read test_num

if [ $test_num -eq 1 ];
then
        echo "./testing_images/test$test_num.jpeg" | python findButtons.py 
else
        echo "./testing_images/test$test_num.jpg" | python findButtons.py 
        
fi
 
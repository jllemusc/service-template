#1st 
countries =['Australia','United Kingdom','Italy','Spain','France','Sweden','Switzerland']
import functions


print(functions.filter_countries(countries,'S'))
print(functions.get_first_country(countries))

print(functions.suma(3,4))
print(functions.division(10,5))

#2nd method

from functions import filter_countries
from functions import get_first_country
from functions import suma
print(filter_countries(countries,'S'))
print(get_first_country(countries))
print(suma(6,6))

#3th method
import sys
sys.path.append('/Users/leo9l/Internship/')

from functions import filter_countries
from functions import get_first_country
from functions import suma
print(filter_countries(countries,'S'))
print(get_first_country(countries))
print(suma(6,6))

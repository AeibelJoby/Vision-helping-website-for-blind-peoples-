from django.db import models

# Create your models here.
class c_reg(models.Model) :
    F_name=models.CharField(max_length=200)
    Email=models.CharField(max_length=30)
    Password=models.CharField(max_length=20)
    def __str__(self):
        return self.F_name
    
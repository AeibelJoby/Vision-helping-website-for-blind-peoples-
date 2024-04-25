from django.urls import path
from . import views
urlpatterns=[
    path('',views.home,name='home'),
    path('log',views.log,name='log'),
    path('reg',views.reg,name='reg'),
    path('home1',views.home1,name='home1'),
    path('addface',views.addface,name='addface'),
    path('fverify',views.face_verify,name='fverify'),
    path('objdec',views.objdec,name='objdec'),


]
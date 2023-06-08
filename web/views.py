from django.shortcuts import render
from ..ml.park import mach
# Create your views here.


def input(request):
    if request.method == 'POST':

        age = request.POST.get('age')
        sex = request.POST.get('sex')
        jitter = request.POST.get('Jitter')
        rap = request.POST.get('RAP')
        ppq = request.POST.get('PPQ')
        ddp = request.POST.get('DDP')
        shimmer = request.POST.get('Shimmer')
        fohz = request.POST.get('Fo(Hz)')
        fhihz = request.POST.get('Fhi(Hz)')
        flohz = request.POST.get('Flo(Hz)')





        
        pass
    return render(request,'input.html')
import pandas
import numpy

__Data = {}

def __createSamples():
  N = 10000

  sm = [0, 0, 0, 0, 0]
  scov = [ [0.1,   0,   0,  0,  0],
           [0,     1, 0.9,  0,  0],
           [0,   0.9,   1,  0,  0],
           [0,     0,   0,  1,  0],
           [0,     0,   0,  0,  1]]


  bm = [0.2, 0, 0, 0, 0]
  bcov = [ [0.1,    0,    0,    0,    0],
           [0,      1, -0.9,    0,  0.1],
           [0,   -0.9,    1,  0.1, -0.1],
           [0,      0,  0.1,    1,  0.1],
           [0,    0.1, -0.1,  0.1,    1]]

  signal = {'var1':[], 'var2':[], 'var3':[], 'var4':[], 'var5':[],}
  background = {'var1':[], 'var2':[], 'var3':[], 'var4':[], 'var5':[],}

  sd = numpy.random.multivariate_normal(sm, scov, N)
  for v1,v2,v3,v4,v5 in sd:
    signal['var1'].append(v1)
    signal['var2'].append(v2)
    signal['var3'].append(v3)
    signal['var4'].append(v4)
    signal['var5'].append(v5)

  bd = numpy.random.multivariate_normal(bm, bcov, N)
  for v1,v2,v3,v4,v5 in bd:
    background['var1'].append(v1)
    background['var2'].append(v2)
    background['var3'].append(v3)
    background['var4'].append(v4)
    background['var5'].append(v5)

  __Data['Signal']     = pandas.DataFrame(signal)
  __Data['Background'] = pandas.DataFrame(background)


def getSignalData():
  if not 'Signal' in __Data:
    __createSamples()

  return __Data['Signal']
  
  
def getBackgroundData():
  if not 'Background' in __Data:
    __createSamples()

  return __Data['Background']

def getExampleDataSet():
  if not 'example' in __Data:

    data = []
    Nsignal = 1000
    Nbackground = 10000
    min_mass = 5000
    max_mass = 6000

    for i in range(Nsignal):
      mass = -1
      while not min_mass <= mass < max_mass:
        mass = numpy.random.normal(5280, 20)
      data.append(mass)

    for i in range(Nbackground):
      mass = -1
      while not min_mass <= mass < max_mass:
        mass = numpy.random.exponential(1e3) + min_mass
      data.append(mass)

    numpy.random.shuffle(data)
    __Data['example'] = pandas.DataFrame({'B invariant mass':data})
    
  return __Data['example']

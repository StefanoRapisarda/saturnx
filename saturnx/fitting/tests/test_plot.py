import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use("TkAgg")
import numpy as np
from ..plot import extractData

class TestPlot:

    def test_extractData(self):
        x = np.linspace(1,10,10)
        y = x.copy()
        np.random.shuffle(y)
        ys = np.random.random(10)
        xs = np.random.random(10)

        fig1 = plt.figure(figsize=(6,6))
        plt.errorbar(x,y,xerr=xs,yerr=ys)

        cx,cy,cxs,cys = extractData(fig1)
        print(cxs-xs)
        assert np.allclose(cx,x)
        assert np.allclose(cy,y)
        assert np.allclose(cxs,xs)
        assert np.allclose(cys,ys)
        
        fig2 = plt.figure(figsize=(6,6))
        plt.errorbar(x,y,xerr=xs)

        cx,cy,cxs,cys = extractData(fig2)
        assert np.allclose(cx,x)
        assert np.allclose(cy,y)
        assert np.allclose(cxs,xs)
        assert cys == None

        fig3 = plt.figure(figsize=(6,6))
        plt.errorbar(x,y,yerr=ys)

        cx,cy,cxs,cys = extractData(fig3)
        assert np.allclose(cx,x)
        assert np.allclose(cy,y)
        assert cxs == None
        assert np.allclose(cys,ys)

        fig4 = plt.figure(figsize=(6,6))
        plt.plot(x,y)

        cx,cy,cxs,cys = extractData(fig4)
        assert np.allclose(cx,x)
        assert np.allclose(cy,y)
        assert cxs == None
        assert cys == None






# -*- coding: utf-8 -*-
'''
Para importar:

!git clone https://github.com/michelleaguirrebuo/eda_lib.git
import sys
sys.path.append('/content/eda_lib')
from eda_lib import EDA
'''




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

#Esta clase es solo para graficar
class RadarHeatmap:
    def __init__(self, df: pd.DataFrame, columns: list, group_col: str = None, ranges=None):
        """
        A radar-style heatmap showing feature-wise density distributions.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        columns : list of str
            Numeric columns to visualize on the radar axes.
        group_col : str, optional
            Name of a categorical column for grouping. If None or missing,
            the radar will plot all data together.
        """
        self.df = df.copy()
        self.columns = columns
        self.group_col = group_col if group_col in df.columns else None
        self.ranges=ranges

        # Validate column existence
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # ---------------------------------------------------
    def plot(self, bw=0.2, alpha=0.5, figsize=(9, 9)):
        """
        Draw the radar heatmap.

        Parameters
        ----------
        bw : float
            Bandwidth for the Gaussian KDE (smaller = more detailed).
        alpha : float
            Transparency level for heat layers (use <1 for overlapping groups).
        figsize : tuple
            Figure size.
        """
        features = self.columns
        num_vars = len(features)

        # Determine grouping
        if self.group_col is None or self.df[self.group_col].isna().all():
            groups = ["All Data"]
        else:
            groups = self.df[self.group_col].dropna().unique()

        # Set up angular coordinates
        theta = np.linspace(0, 2 * np.pi, num_vars + 1)

        # Figure setup
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels(features, fontsize=12)
        ax.set_yticklabels([])
        ax.tick_params(axis='x', direction='out', pad=50)

        # Colormaps for different groups
        cmaps = ["plasma", "viridis", "magma", "cividis", "inferno"]

        # --- Loop over groups ---
        for g_i, group in enumerate(groups):
            sub = (
                self.df[self.df[self.group_col] == group]
                if self.group_col
                else self.df
            )

            densities = []
            radial_grids = []

            # Compute KDE for each feature
            for feature in features:
                data = sub[feature].dropna()
                if len(data) < 2:
                    continue
                kde = gaussian_kde(data, bw_method=bw)
                r_feat = np.linspace(data.min(), data.max(), 300)
                dens = kde.evaluate(r_feat)
                dens = dens / dens.max()  # normalize per feature
                densities.append(dens)
                radial_grids.append(r_feat)

            # Close the loop (wrap around)
            densities.append(densities[0])
            radial_grids.append(radial_grids[0])

            # Normalize each axis’s scale to 0–1 for display
            max_len = max(len(r) for r in radial_grids)
            r_norm = np.linspace(0, 1, max_len)
            Z = np.zeros((num_vars + 1, max_len))
            for i in range(num_vars + 1):
                r_i = radial_grids[i]
                dens_i = densities[i]
                r_i_norm = (r_i - r_i.min()) / (r_i.max() - r_i.min())
                Z[i] = np.interp(r_norm, r_i_norm, dens_i)

            # Create polar heat mesh
            R, T = np.meshgrid(r_norm, theta)
            ax.pcolormesh(
                T,
                R,
                Z,
                cmap=cmaps[0],
                shading="auto",
                alpha=alpha,
                label=str(group),
            )


            # Add min–max labels per feature
        for i, feature in enumerate(features):
                txt= str(self.ranges) if self.ranges else f"{data.min():.2f}\n|\n{data.max():.2f}"
                data = self.df[feature]
                ax.text( theta[i], 1.10, txt, ha="center", va="center", fontsize=8, color="gray", )



        # Legend only if groups exist
        if self.group_col and len(groups) > 1:
            plt.legend(
                loc="upper right", bbox_to_anchor=(1.25, 1.1), title=self.group_col
            )


        plt.tight_layout()
        plt.show()





class EDA:
    '''
    Exploratory Data Analysis Library
    '''
    def __init__(self, file:str='', sheet:str or int=0, set_df:bool=False, df=None):
        '''
        ### Description
        Opens file
        ### Params
        - file: File name in string format.
        - sheet: 0 by default. Sheet name in case of an excel file w/
          multiple sheets
        '''
        from pandas import read_excel,read_csv
        import warnings
        warnings.filterwarnings('ignore')
        d={'xlsx':read_excel,'csv':read_csv}
        ext=file.split('.')[-1] if not set_df else None
        self.file_name=file if not set_df else None
        if set_df: self.df=df
        else:
          try:
              openf=d[ext]
              print('File format correct.')
          except KeyError:
              print('Unreadable file format. Try a csv file or an xlsx file instead.')
          try:
              if ext != 'csv':
                  self.df=openf(file,sheet_name=sheet)
                  print('File opened succesfully.')
              else:
                  with open(file) as f:
                      self.df=openf(f, encoding="utf-8")
                      print('File opened succesfully.')
          except FileNotFoundError:
              print('File path not found')
          else:
              pass
        cols=self.df.columns
        for col in cols:
            self.df.rename({col:col.lower().removeprefix('radarpsychometric').removesuffix('psychometric')},axis=1,inplace=True)

    def visualize(self,columns:list or str=[],rows=5, head=True, tail=False):
        '''
        ### Description
        Returns the first or last N columns of the dataset
        ### Params
        - columns: [] by default. List or str containing columns to show.
        - rows: 5 by default. Number of rows to show
        - head: True by default. return N *first* columns
        - tail: False by default. Return N *last* columns
        '''
        df=self.df[columns] if columns else self.df
        if head:
            return df.head(rows)
        else:
            return df.tail(rows)

    def describe(self, columns:list or str=[]):
        '''
        ### Description
        Descriptive analysis of numeric columns of the dataset.
        Does not ignore NaNs or NaN-filled columns.
        ### Params
        - columns: [] by default. List or str containing columns to show.
        '''
        if not columns:
            return self.df.describe()
        else:
            return self.df[columns].describe()

    def columns(self):
        '''
        ### Description
        Returns columns currently contained in the dataset
        '''
        return self.df.columns

    def index(self):
        '''
        ### Description
        Returns index of the current dataset
        '''
        return self.df.index

    def reset_index(self,drop_index:bool=True):
        '''
        ### Description
        Resets index of the dataframe inplace.
        ### Params
        - drop: True by default. Bool indicating whether to
        drop the current index
        '''
        self.df.reset_index(inplace=True,drop=drop_index)

    def info(self, columns: list or str=[]):
        '''
        ### Description
        Returns info on count and datatypes of columns
        ### Params
        - columns: [] by default. List or str containing columns to show.
        '''
        df=self.df[columns] if columns else self.df
        return df.info()

    def missing_proportion(self, columns: list or str=[]):
        '''
        ### Description
        Returns proportion of missing values over determined columns
        ### Params
        - columns: [] by default. List or str containing columns to show.
        '''
        df=self.df[columns] if columns else self.df
        return df.isna().sum()/self.df.shape[0]*100

    def duplicate_proportion(self, columns: list or str=[]):
        '''
        ### Description
        Return proportion of duplicated values over given column or
        prints proportion of duplicated values over list of columns.
        Ignores missing values.
        ### Params
        - columns: [] by default. List or str containing columns to show.
        '''
        df=self.df[~self.df[columns].isna()][columns]if columns else self.df
        if type(columns)!=str:
          for i in df.columns:
              print(f'{i:<40}{df[~df[i].isna()].duplicated(subset=i,keep=False).sum()/self.df.shape[0]*100:>60.3f}')
        else:
          return df.duplicated().sum()/self.df.shape[0]*100

    def duplicates_count(self, columns: list or str=[]):
        '''
        ### Description
        Return proportion of duplicated values over given column or
        prints proportion of duplicated values over list of columns.
        Ignores missing values.
        ### Params
        - columns: [] by default. List or str containing columns to show.
        '''
        df=self.df[~self.df[columns].isna()][columns]if columns else self.df
        if type(columns)!=str:
          for i in df.columns:
              print(f'{i:<40}{df[~df[i].isna()].duplicated(subset=i,keep=False).sum():>60.3f}')
        else:
          return df.duplicated().sum()

    def duplicates(self, columns: list or str=[]):
        '''
        ### Description
        Return  duplicated values over given column or
        prints duplicated values over list of columns.
        Ignores missing values.
        ### Params
        - columns: [] by default. List or str containing columns to show.
        '''
        df=self.df[~self.df[columns].isna()][columns]if columns else self.df
        if type(columns)!=str:
          for i in df.columns:
              print(f'{i}\n{df[~df[i].isna()].duplicated(subset=i,keep=False)}\n')
        else:
          return df.duplicated(subset=columns,keep=False)

    def value_counts(self, columns:list or str=[], ignore_nans: bool=False):
        '''
        ### Description
        Returns unique value counts by column
        ### Params
        - columns: [] by default. List containing string formatted
          names of columns to describe
        - ignore_nans: False by default. Include nans in unique value count or not
        '''
        if type(columns)==str:
          return self.df[columns].value_counts(dropna=ignore_nans)
        else:
          for col in columns:
            print(self.df[col].value_counts(dropna=ignore_nans))

    def uniques_proportion(self, columns:list or str=[], ignore_nans: bool=False):
        '''
        ### Description
        Returns unique value proportions by column
        ### Params
        - columns: [] by default. List containing string formatted
          names of columns to describe
        - ignore_nans: False by default. Include nans in unique value count or not
        '''
        if type(columns)==str:
          return self.df[columns].value_counts(normalize=True,dropna=ignore_nans)*100
        else:
          for col in columns:
            print(self.df[col].value_counts(normalize=True,dropna=ignore_nans)*100)


    def uniques(self, columns:list or str=[]):
        return self.df[columns].unique() if type(columns)==str else [(col,self.df[col].unique() )for col in columns]


    def labelEncoding(self, column:str, ignore_nans=True, customOrder:list=None):
        '''
        ### Description
        Encodes unique values of a column as unique sequential integer values
        ### Params
        - column: Name of a column in string format
        - ignore_nans: True by default. Consider NaNs unique values of the column
        '''
        from numpy import nan
        if not customOrder:
          unq=self.df[column].unique() if ignore_nans else self.df[~self.df[column].isna()][column].unique()
        else:
          unq=customOrder+[nan] if not ignore_nans else customOrder
        dct={u:idx for idx,u in enumerate(unq)}
        self.df[column+'_encoded'] = self.df[column].map(dct)

    def oneHotEncoding(self, column:str, ignore_nans=True, customOrder:list=None):
        '''
        ### Description
        Encodes unique values of a column as binary values in sequential columns
        with the name of the uique value encoded
        ### Params
        - column: Name of a column in string format
        - ignore_nans: True by default. Consider NaNs unique values of the column
        '''
        from numpy import nan
        if not customOrder:
          unq=self.df[column].unique() if ignore_nans else self.df[~self.df[column].isna()][column].unique()
        else:
          unq=customOrder+[nan] if not ignore_nans else customOrder
          for u in unq:
            dct={ui:(1 if ui==u else 0) for ui in unq}
            self.df[u]=self.df[column].map(dct)

    def edad(self,column:str, fecha_corte:str='', fmt:str='Y', returnColumn:str='Edad'):
        '''
        ### Description
        Create an age column based on difference between the birth date of an
        individual and a given date. If no date is given, today´s date considered
        as fecha_corte.
        ### Params
        - column: Name of a column in string format. This is the birth date
        - fecha_corte: a date in string format. This parameter is optional.
        - fmt: "Y" by default. The format of the return column.
          "Y","M","D","W" are other formatting options.
        - returnColumn: "Edad" by default. Name of the returned column.
        '''
        from pandas import to_datetime,Series
        from datetime import datetime
        from numpy import timedelta64
        self.df[column].apply(lambda x: to_datetime(x))
        y=Series([datetime.now() if not fecha_corte else to_datetime(fecha_corte) for _ in range(self.df[column].shape[0])])
        self.df[returnColumn]=(y-self.df[column])/timedelta64(1, fmt)

    def antiguedad(self,column:str, fecha_corte:str='', fmt:str='Y', returnColumn:str='Antiguedad'):
        '''
        ### Description
        Create a seniority column based on difference between the entry date of an
        individual and a given date. If no date is given, today´s date considered
        as fecha_corte.
        ### Params
        - column: Name of a column in string format. This is the birth date
        - fecha_corte: a date in string format. This parameter is optional.
        - fmt: "Y" by default. The format of the return column.
          "Y","M","D","W" are other formatting options.
        - returnColumn: "Antiguedad" by default. Name of the returned column.
        '''
        from pandas import to_datetime,Series
        from datetime import datetime
        from numpy import timedelta64
        self.df[column].apply(lambda x: to_datetime(x))
        y=Series([datetime.now() if not fecha_corte else to_datetime(fecha_corte) for _ in range(self.df[column].shape[0])])
        self.df[returnColumn]=(y-self.df[column])/timedelta64(1, fmt)

    def dropColumns(self, columns:list=[]):
        self.df.drop(columns,axis=1,inplace=True)

    def dropRows(self, rows:list=[]):
        self.df.drop(rows, axis=0, inplace=True)

    def dropDuplicates(self, columns:list):
        self.df.drop_duplicates(subset=columns, inplace=True)

    def dropNaNs(self, columns:list or str, na=None):
        '''
        ### Params
        - columns: list of columns in string format
          or column name in string format with NaNs
          to be dropped from the dataframe
        - na: None by default. Replace value if NaNs
          are encoded in any way other than np.nan
        '''
        from numpy import nan
        if na in [None]:
          self.df.dropna(subset=columns, inplace=True)
        else:
          for c in self.df.columns:
            self.df.loc[self.df[c]==na, columns]=nan
          self.df.dropna(subset=columns, inplace=True)

    def fillNaNs(self, column:str, value):
        self.df[column].fillna(value,inplace=True)

    def proportionByGroup(self, columns:list or str, ignore_nans=False):
        return self.df.groupby(columns).value_counts(normalize=True,dropna=ignore_nans)

    def countByGroup(self, columns:list or str, ignore_nans=False):
        return self.df.groupby(columns).value_counts(dropna=ignore_nans)

    def addTruncatedColumn(self, column:str, truncateAt:int=11, last:bool=False):
        df=self.df[column].apply(lambda x: str(x))
        self.df[column+f'_{'last' if last else 'First'}{truncateAt}']=df.apply(lambda x: x[:truncateAt])
        return self.df[column+f'_{'last' if last else 'First'}{truncateAt}']

    def dispersion(self, columns: list or str):
        from pandas import DataFrame
        describe=self.df[columns].describe().loc[['std','max','min','25%','75%','mean','50%']]
        disp=DataFrame()
        disp['std']=describe.loc['std'].T
        disp['iqr']=describe.loc['75%'].T-describe.loc['25%'].T
        disp['range']=describe.loc['max'].T-describe.loc['min'].T
        disp['IQR/Range']=disp['iqr']/disp['range']
        disp['CV']=(disp['std']/describe.loc['mean'])*100
        disp['EDA_median']=(self.df[columns]-describe.loc['50%']).median().T
        return disp

    def normalize(self, columns: list or str, custom_scale: tuple = (None, None)):
        if custom_scale!=(None, None):
          min, max = custom_scale
          self.df[columns]=(self.df[columns]-min)/(max - min)
        else:
          self.df[columns]= (self.df[columns]-self.df[columns].min())/(self.df[columns].max()-self.df[columns].min())

    def rangosPsicometrico(self,df=None,set_df=False,nec:str='perfil', disp_metric:str= 'CV', varsFromCuadrant=5, divs=12, printRanges=False, kde=False, spiderweb=False, ):
      from seaborn import kdeplot
      from matplotlib.pyplot import figure,xlim,title
      c1=['iniciativa', 'inteligenciasocial','influencia', 'autonomia']
      c2=['desarrollo', 'orientacionservicio','diplomacia', 'disponibilidad']
      c3=['precision', 'atencionfocalizada','pensamientoanalitico', 'exctecnica']
      c4=['implementacion', 'expeditividad','determinacion', 'agentecambio']
      repna=['r', 'e','p', 'n', 'a']
      cuads=(c1,c2,c3,c4,repna)
      labels = {0:'Cuadrante 1', 1:'Cuadrante 2', 2:'Cuadrante 3', 3:'Cuadrante 4', 4:'Repna'}
      ic=[]
      everything=[]
      for cuadrant in cuads:
        if disp_metric:
          ic.append(self.dispersion(cuadrant).sort_values(disp_metric).round().head(varsFromCuadrant).index)
        else:
          ic.append(self.dispersion(cuadrant).round().index)
      for var,cuadrant in enumerate(ic):
        if printRanges:print(labels[var])
        for column in cuadrant:
          x=self.df.loc[~self.df[column].isna(),column] if not set_df else df.loc[~df[column].isna(),column]
          step=int(100/divs)
          sums=[]
          for i in range(0,100,step):
            upper_b=i+step if i+step<100 else 101
            counts=x.value_counts(normalize=True)*100
            consult=counts.loc[(counts.index<upper_b) & (counts.index>=i)]
            sums.append(((i,upper_b-1),(consult.sum())))
          sums2=[i[1].round(1) for i in sums]
          idxs=[i[0] for i in sums]
          if kde:figure(),kdeplot(x,bw_adjust=0.8),title(column),xlim(0,100)
          if printRanges:
            print(column,index:=max(sums2),idxs[sums2.index(index)])
          everything.append((column,sums))
      if spiderweb:
        self.graficarPsicometrico([col for cuadrante in cuads for col in cuadrante],nec)
      return everything


    def concentracionRangosPsico(self, rangos:list):
      from numpy import cumsum,nan,diff,round
      from pandas import DataFrame
      cols=[i[0] for i in rangos]
      rans=[list(list(zip(*i[1]))[0]) for i in rangos]
      mx=max([len(ran) for ran in rans])
      conc=[list(list(zip(*i[1]))[1]) for i in rangos]
      df=DataFrame()
      for idx,col in enumerate(cols):
        df[col+'_rangos']=rans[idx] if (ln:=len(rans[idx]))==mx else rans[idx]+[nan for _ in range(mx-ln)]
        df[col+'_concentración']=cumsum(conc[idx]) if (ln:=len(rans[idx]))==mx else cumsum(conc[idx]+[nan for _ in range(mx-ln)])
        df[col+'_diff']=diff(df[col+'_concentración'], prepend=0)
        df[col+'_concentración'],df[col+'_diff']=df[col+'_concentración'].round(2),df[col+'_diff'].round(2)
      return df

    def concentracionDiferencialGlobal(self, performanceCol:str, categories:list ,nec:str='perfil', divs=12, spiderweb=False, plots=False):
      from pandas import DataFrame,concat
      from matplotlib.pyplot import figure,xlim,title,xticks,figlegend, plot,title
      c1=['iniciativa', 'inteligenciasocial','influencia', 'autonomia']
      c2=['desarrollo', 'orientacionservicio','diplomacia', 'disponibilidad']
      c3=['precision', 'atencionfocalizada','pensamientoanalitico', 'exctecnica']
      c4=['implementacion', 'expeditividad','determinacion', 'agentecambio']
      repna=['r', 'e','p', 'n', 'a']
      cuads=(c1,c2,c3,c4,repna)
      df=self.df.loc[((self.df[performanceCol]==categories[0])|(self.df[performanceCol]==categories[1])),:]
      N=len(df)
      cat1= df.loc[(df[performanceCol]==categories[0])]
      cat2= df.loc[(df[performanceCol]==categories[1])]
      result=DataFrame()
      step=int(100/divs)
      for cuadrant in cuads:
        for variable in cuadrant:
          aux = []
          for i in range(0, 100, step):
              lb = i
              ub = i + step if i + step < 100 else 101
              ran = (lb, ub - 1)
              totCon = df.loc[(df[variable]>=lb)&(df[variable]<ub), variable].count() / N * 100
              cat1Con = cat1.loc[(cat1[variable]>=lb)&(cat1[variable]<ub), variable].count() / N * 100
              cat2Con = cat2.loc[(cat2[variable]>=lb)&(cat2[variable]<ub), variable].count() / N * 100
              diffCon = abs(cat1Con - cat2Con)
              aux.append([ran, totCon, cat1Con, cat2Con, diffCon])
          tmp = DataFrame(
              aux,
              columns=[
                  f"range_{variable}",
                  f"total_con_{variable}",
                  f"cat1_con_{variable}",
                  f"cat2_con_{variable}",
                  f"diff_con_{variable}"
              ]
          )
          result = concat([result, tmp], axis=1)
      if spiderweb:
        radar=RadarHeatmap(result.loc[:,result.columns.str.contains('diff')],columns=result.loc[:,result.columns.str.contains('diff')].columns)
        radar.plot(bw=0.1)
      if plots:
        dfx=result.loc[:,result.columns.str.contains('cat')]
        ran=result.loc[:,result.columns.str.contains('range')][result.loc[:,result.columns.str.contains('range')].columns[0]]
        ln=len(dfx.columns)
        columns=dfx.columns
        for i in range(0,ln,2):
          plt.figure()
          plt.plot(dfx[columns[i]],c='r',label=columns[i])
          plt.plot(dfx[columns[i+1]],c='b',label=columns[i+1])
          plt.xticks([i for i in range(len(ran))],ran, rotation=90)
          plt.figlegend()

      return result

    def concentracionDiferencialRelativa(self, performanceCol:str, categories:list or tuple, divs=12, spiderweb=False, plots=False):
        from pandas import DataFrame,concat
        from matplotlib.pyplot import figure,xlim,title,xticks,figlegend, plot
        rangos= [self.rangosPsicometrico(df=self.df[self.df[performanceCol]==categories[0]],divs=divs,set_df=True),
                 self.rangosPsicometrico(df=self.df[self.df[performanceCol]==categories[1]],divs=divs,set_df=True)]
        conc= [self.concentracionRangosPsico(rangos[0]),self.concentracionRangosPsico(rangos[1])]
        x= concat([conc[0].loc[:,conc[0].columns.str.contains('diff')],conc[1].loc[:,conc[1].columns.str.contains('diff')]],axis=0)
        x['cat']=[categories[0]]*len(conc[0])+[categories[1]]*len(conc[1])
        df=DataFrame()
        diffCols=[]
        cols=[i[0] for i in rangos[0]]
        for idx,col in enumerate(cols):
          df[col+'_rangos']=conc[0].loc[:,col+'_rangos']
          y= [r[0] for r in df[col+'_rangos']]
          df[col+'_diff']=conc[0].loc[:,col+'_diff']-conc[1].loc[:,col+'_diff']
          df[col+'_'+categories[0]+'_diff']=conc[0].loc[:,col+'_diff']
          df[col+'_'+categories[1]+'_diff']=conc[1].loc[:,col+'_diff']
          if plots:
            figure()
            plot(y,x.loc[x['cat']==categories[0],col+'_diff'], c='r', label=categories[0])
            plot(y,x.loc[x['cat']==categories[1],col+'_diff'], c='b', label=categories[1])
            xticks(y,conc[0].loc[:,col+'_rangos'],rotation=90)
            title(col)
            xlim(-1,100)
            figlegend()
          diffCols.append(col+'_diff')
        if spiderweb:
          radar=RadarHeatmap(df=df,columns=diffCols, ranges=(0,100))
          radar.plot(bw=0.1)
        return df

    def graficarPsicometrico(self, columns:list, nec:str='perfil'):
        radar=RadarHeatmap(self.df[~self.df[nec].isna()] , columns=columns)
        radar.plot(bw=0.1)

    def topPerfiles(self, rows:int=5, superpoder:bool=True, ignore_nans:bool=True):
        cols=['perfil'] if not superpoder else ['perfil','superpoder']
        cons=self.df[cols].value_counts(dropna=ignore_nans).reset_index(drop=False)
        cons['%']=self.df[cols].value_counts(dropna=ignore_nans,normalize=True).reset_index(drop=False)['proportion'].round(4)*100
        return cons.head(rows)

    def avg_median(self,columns: list or str,nec:str='perfil'):
        df=self.df[~self.df['perfil'].isna()][columns]
        return df.describe().T[['mean','50%']]

    def low_mid_high_proportions(self):
      from seaborn import kdeplot
      from matplotlib.pyplot import figure,xlim,title
      cols_psico=[ 'r', 'e', 'p', 'n', 'a',
       'iniciativa', 'inteligenciasocial', 'influencia', 'autonomia',
       'desarrollo', 'orientacionservicio', 'diplomacia', 'disponibilidad',
       'precision', 'atencionfocalizada', 'pensamientoanalitico', 'exctecnica',
       'implementacion', 'expeditividad', 'determinacion', 'agentecambio']
      everything=[]
      for column in cols_psico:
        x=self.df.loc[~self.df[column].isna(),column] 
        step=33
        sums=[]
        for i in range(0,98,step):
              upper_b=i+step if i+step<98 else 101
              counts=x.value_counts(normalize=True)*100
              consult=counts.loc[(counts.index<upper_b) & (counts.index>=i)]
              sums.append(((i,upper_b-1),(consult.sum())))
        sums2=[i[1].round(1) for i in sums]
        idxs=[i[0] for i in sums]
        everything.append((column,sums))
      return self.concentracionRangosPsico(everything)



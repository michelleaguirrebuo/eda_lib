# -*- coding: utf-8 -*-
'''
Para importar:

!git clone https://github.com/michelleaguirrebuo/eda_lib.git
import sys
sys.path.append('/content/eda_lib')
import eda_lib
'''




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

#Esta clase es solo para graficar
class RadarHeatmap:
    def __init__(self, df: pd.DataFrame, columns: list, group_col: str = None):
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
        for i, tick_label in enumerate(ax.get_xticklabels()):
        # Set rotation based on index, for example, increasing rotation
            tick_label.set_rotation(360/(i+1))
        ax.tick_params(axis='x' direction='out', pad=50)

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
                cmap=cmaps[g_i % len(cmaps)],
                shading="auto",
                alpha=alpha,
                label=str(group),
            )

        # Add min–max labels per feature
        for i, feature in enumerate(features):
            data = self.df[feature]
            ax.text(
                theta[i],
                1.1,
                f"{data.min():.2f}\n|\n{data.max():.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
            )

        # Legend only if groups exist
        if self.group_col and len(groups) > 1:
            plt.legend(
                loc="upper right", bbox_to_anchor=(1.25, 1.1), title=self.group_col
            )

        plt.colorbar(ax.collections[0], label="Density (normalized)", shrink=0.8)
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

    def reindex(self,drop: bool =True):
        '''
        ### Description
        Resets index of the dataframe inplace.
        ### Params
        - drop: True by default. Bool indicating whether to
        drop the current index
        '''
        self.df.reset_index(inplace=True,drop=drop)

    def info(self, columns: list or str=[]):
        '''
        ### Description
        Returns info on count and datatypes of columns
        ### Params
        - columns: [] by default. List or str containing columns to show.
        '''
        df=self.df[columns] if columns else self.df
        return df.info()

    def missingValsProportion(self, columns: list or str=[]):
        '''
        ### Description
        Returns proportion of missing values over determined columns
        ### Params
        - columns: [] by default. List or str containing columns to show.
        '''
        df=self.df[columns] if columns else self.df
        return df.isna().sum()/self.df.shape[0]*100

    def duplicateValsProportion(self, columns: list or str=[]):
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

    def duplicateValsCount(self, columns: list or str=[]):
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

    def duplicateVals(self, columns: list or str=[]):
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

    def contarValsUnicos(self, columns:list or str=[], ignore_nans: bool=False):
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

    def contarProporcionValsUnicos(self, columns:list or str=[], ignore_nans: bool=False):
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


    def valsUnicos(self, columns:list or str=[]):
        return self.df[columns].unique() if type(columns)==str else [(col,self.df[col].unique() )for col in columns]


    def labelEncoding(self, column:str, ignore_nans=True):
        '''
        ### Description
        Encodes unique values of a column as unique sequential integer values
        ### Params
        - column: Name of a column in string format
        - ignore_nans: True by default. Consider NaNs unique values of the column
        '''
        unq=self.df[column].unique() if ignore_nans else self.df[~self.df[column].isna()][column].unique()
        dct={u:idx for idx,u in enumerate(unq)}
        self.df[column+'_encoded'] = self.df[column].map(dct)

    def oneHotEncoding(self, column:str, ignore_nans=True):
        '''
        ### Description
        Encodes unique values of a column as binary values in sequential columns
        with the name of the uique value encoded
        ### Params
        - column: Name of a column in string format
        - ignore_nans: True by default. Consider NaNs unique values of the column
        '''
        unq=self.df[column].unique() if ignore_nans else self.df[~self.df[column].isna()][column].unique()
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

    def dropDups(self, columns:list):
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

    def propByGroup(self, columns:list or str, ignore_nans=False):
        return self.df.groupby(columns).value_counts(normalize=True,dropna=ignore_nans)

    def countByGroup(self, columns:list or str, ignore_nans=False):
        return self.df.groupby(columns).value_counts(dropna=ignore_nans)

    def createExcel(self):
        self.df.to_excel('output'+self.file_name,index=False)

    def createCSV(self):
        self.df.to_csv('output'+self.file_name,index=False,sep=',',na_rep='N/A', encoding="utf-8", header=True)

    def merge(self, df_other, columns: list or str, join_type='left', keep=False, suffixes=('','')):
        merged = self.df.merge(df_other, on=columns, how=join_type,suffixes=suffixes)
        if keep: self.df=merged 
        return merged

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

    def rangosPsicometrico(self, disp_metric:str= 'CV', useN_fromCuad=2, divPerRange=12, printRanges=False, graph=False):
      from seaborn import kdeplot
      from matplotlib.pyplot import figure,xlim,title
      c1=['radarpsychometriciniciativa', 'radarpsychometricinteligenciasocial','radarpsychometricinfluencia', 'radarpsychometricautonomia']
      c2=['radarpsychometricdesarrollo', 'radarpsychometricorientacionservicio','radarpsychometricdiplomacia', 'radarpsychometricdisponibilidad']
      c3=['radarpsychometricprecision', 'radarpsychometricatencionfocalizada','radarpsychometricpensamientoanalitico', 'radarpsychometricexctecnica']
      c4=['radarpsychometricimplementacion', 'radarpsychometricexpeditividad','radarpsychometricdeterminacion', 'radarpsychometricagentecambio']
      repna=['rpsychometric', 'epsychometric','ppsychometric', 'npsychometric', 'apsychometric']
      cuads=(c1,c2,c3,c4,repna)
      labels = {0:'C1', 1:'C2', 2:'C3', 3:'C4', 4:'Repna'}
      ic=[]
      everything=[]
      for cuadrant in cuads:
        if disp_metric:
          ic.append(self.dispersion(cuadrant).sort_values(disp_metric).round().head(useN_fromCuad).index)
        else:
          ic.append(self.dispersion(cuadrant).round().index)
      for label,cuadrant in enumerate(ic):
        print(labels[label])
        for column in cuadrant:
          df=self.df[~self.df[column].isna()][column]
          step=int((df.max()-df.min())//divPerRange)
          sums=[]
          for i in range(int(df.min()),int(df.max()),step):
            upper_b=i+step if i+step<df.max() else int(df.max()+1)
            bit1=((df.value_counts().index<(upper_b)))
            bit2=df.value_counts().index>=i
            idxs= [b1 & b2 for b1,b2 in zip(bit1,bit2)]
            sums.append(((i,upper_b-1),((((df.value_counts().loc[idxs]))/df.value_counts().sum()*100).sum())))
          sums2=[i[1].round(1) for i in sums]
          idxs=[i[0] for i in sums]
          print(column,index:=max(sums2),idxs[sums2.index(index)])
          if graph:figure(),kdeplot(self.df[column],bw_adjust=0.8),title(column),xlim(0,100)
          if printRanges:print(sums)
          everything.append((column,sums))
      return everything

    def graficarPsicometrico(self, columns:list, group_col=None):
        radar=RadarHeatmap(self.df, columns=columns, group_col=group_col)
        radar.plot(bw=0.2)

    def normalityTest(self,columns: list or str):
      from scipy.stats import shapiro
      return shapiro(self.df[columns].T,axis=1).statistic

    def bimodalityCheck(self,columns: list or str):
      from scipy.stats import kurtosis
      from pandas import DataFrame
      g2 = self.df[columns].skew()
      g3 = kurtosis(self.df[columns],fisher=False)
      BC = (g3 + 1) / (g2**2 + 3)
      df=DataFrame(BC).T
      df.columns=columns
      return df

    def concentracionRangosPsico(self, rangos:list):
      from numpy import cumsum,nan,diff
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
      return df

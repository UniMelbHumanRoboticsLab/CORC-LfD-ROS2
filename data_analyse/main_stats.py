import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_process.file_util_pkg import load_npy
from data_analyse.stats_pkg import assign_stars,remove_outliers_iqr

import numpy as np
np.set_printoptions(suppress=True,precision=4) # suppress scientific notation
import pandas as pd
import matplotlib.pyplot as plt
fontsize = 8
plt.rcParams.update({'font.size': fontsize})

import scipy.stats as stats
import pingouin as pg
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings(action="ignore",module="statsmodels")

cases = ["recon","recon_lut"]
subjects = [
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    [11,12,13,14,15,16,17,18,19,20,21,22,23,24]]

# normality check plot
fig = plt.figure(figsize=((9)*0.393701,7.1*0.393701),num="all stats",layout="constrained")
fig2 = plt.figure(figsize=((9)*0.393701,7.5*0.393701),num="all stats res",layout="constrained")
i = 0

for metric_id,metric_name in zip(["avg_norm_error_mean","norm_diff_tau_mean","coverage_mean"],[r"$\epsilon$",r"$\Delta\tau_{peak}$", r"$C~(\%)$" ]):    
    for eval_id,eval_name in zip(["val","test"],["Seen","Unseen"]):
        # prep the data for stats
        stats_samples = []
        for p in range(1,4):
            for sub in subjects[p-1]:

                session_data = {
                    "exp_id":"exp1_tnsre_trained4",
                    "patient_id":f"p{p}",
                    "subject_id":f"sub{sub}",
                }
                
                subject_path = os.path.join(os.path.dirname(__file__), '..',f'logs/pycorc_recordings/{session_data["exp_id"]}/{session_data["patient_id"]}/{session_data["subject_id"]}')
                patient_subject_samples = load_npy(f"{subject_path}/repro/{eval_id}_processed.npy")
                patient_subject_samples_df = pd.DataFrame(patient_subject_samples)
                
                if len(patient_subject_samples)>0:
                    for method_id,method_name in zip(cases,[f"TPGMM","LUT"]):
                        mean_metric = np.mean(remove_outliers_iqr(np.array(patient_subject_samples_df[f"{method_id}_{metric_id}"].tolist()))[0])
                        stats_sample = {
                            "patient": f"p{p}",
                            "subject": f"sub{sub}",
                            "value": mean_metric,
                            "method":method_name
                            }
                        stats_samples.append(stats_sample)
        stats_df = pd.DataFrame(stats_samples)
        
        # check normality 
        # fit the data in a linear mixed methods model, find the residuals of the fitting, then plot the residuals QQ plot using a normal distribution 
        model = smf.mixedlm("value ~ patient * method", stats_df, groups=stats_df["subject"])
        result = model.fit(reml=True,method="Powell")
        residuals = result.resid
        stats_df["residuals"] = residuals
        
        id = 2*(i+1) - 1 if (i+1) <= 3 else 2*((i+1) - 3)
        
        # plot residual QQ
        ax = fig2.add_subplot(2,3,i+1)
        _,r = stats.probplot(residuals, dist="norm", plot=ax,rvalue=False,fit=True)
        r2 = r[2]**2
        ax.set_title(f'{eval_name}\n{metric_name}\n'+r"$R^2$="+f"{r2:1.2f}")
        ax.set_xlabel("Theoretical")
        ax.set_ylabel("Sample")
        
        # plot estimated distribution
        ax1 = fig.add_subplot(2,3,i+1)
        # Calculate KDE
        kde = stats.gaussian_kde(residuals, bw_method="scott")
        # Get data range
        data_min, data_max = np.min(residuals), np.max(residuals)
        # Get bandwidth for extension
        bandwidth = np.sqrt(kde.covariance[0, 0])
        # Create extended evaluation points (smooth tails like seaborn)
        extend = 0.2 * bandwidth
        x_eval = np.linspace(data_min - extend, data_max + extend, 1000)
        # Evaluate KDE density
        density = kde(x_eval)
        max_density = np.argmax(density)
        # Plot violin (mirrored density for symmetric appearance)
        ax1.fill_between(
            x_eval, 
            0, 
            density,
            facecolor="blue", 
            alpha=0.7, 
            edgecolor='black', 
            linewidth=1.5,
            zorder=2
        )
        ax1.hist(residuals, bins=30, density=True,edgecolor="black",zorder=1)
        ax1.set_xlabel("Residual")
        
        ax1.set_title(f'{eval_name}\n{metric_name}\n'+r"$R^2$="+f"{r2:1.2f}")
       
        if r2 < 0.97:
            print(f'{eval_name} {metric_name}',r2)
            outliers = stats_df[np.abs(residuals) > (np.mean(np.abs(residuals)) + 3 * np.std(np.abs(residuals)))]
            print(outliers[["patient","subject","residuals","method"]])
        
        # 2-way RM Anova
        # check for sphericity
        sphericity = [
            pg.sphericity(dv="value",within="patient",subject="subject",data=stats_df,).spher,
            pg.sphericity(dv="value",within="method",subject="subject",data=stats_df,)[0],
            pg.sphericity(dv="value",within=["patient","method"],subject="subject",data=stats_df,).spher,
            ]
        aov = pg.rm_anova(
                    dv="value",
                    within=["patient", "method"],
                    subject="subject",
                    data=stats_df,
                    detailed=True
                ).round(3)
        aov["spher"] = sphericity
        aov['p_select'] = np.where(aov['spher'], aov['p_unc'], aov['p_GG_corr'])
        assign_stars(aov,"p_select")
        aov["eval"] = eval_id
        aov["metric"] = metric_id
        aov["P-val"] = aov["p_select"].astype(str) + aov["stars"]
        aov.rename(columns={"ddof1": "df1", "ddof2": "df2"}, inplace=True)
        print('==================================')
        print(f'{eval_name} {metric_name}')
        
        # post hocs
        stats_df['cell'] = stats_df['patient'].astype(str) + '_' + stats_df['method'].astype(str)
        post_hoc = pg.pairwise_tests(dv="value", within=['patient','method'], subject='subject',data=stats_df,
                    parametric=True,marginal=True,alpha=0.05,padjust="holm").round(3)
        post_hoc['p_corr'] = post_hoc['p_corr'].fillna(post_hoc['p_unc'])
        assign_stars(post_hoc,"p_corr")
        labels = post_hoc["A"] + "-" + post_hoc["B"]
        post_hoc["Contrast"] = labels
        post_hoc["P-val"] = post_hoc["p_corr"].astype(str) + post_hoc["stars"]
        post_hoc.rename(columns={"dof": "df"}, inplace=True)
        
        # compile all the stats
        col = f"{eval_name} {metric_name}"
        print(aov[["Source","SS","df1","df2","MS","F","P-val"]])
        aovT = aov[["SS","df1","df2","MS","F","P-val"]].T
        # series of functions to restructure the df
        aovT = (
            aovT.reset_index(names="metric")  # make index into a column "metric"
                .melt(id_vars="metric", var_name="src_idx", value_name=f"{eval_name} {metric_name}")
                .assign(Category=lambda d: d["src_idx"].map(aov["Source"]))  # 0,1,2 -> Source names
                .drop(columns="src_idx")
                .rename(columns={"metric": "column"})
        )
        
        print(post_hoc[["Contrast","T","df","P-val"]])
        phT = post_hoc[["T","df","P-val"]].T
        phT = (
            phT.reset_index(names="metric")  # make index into a column "metric"
                .melt(id_vars="metric", var_name="src_idx", value_name=f"{eval_name} {metric_name}")
                .assign(Category=lambda d: d["src_idx"].map(post_hoc["Contrast"]))  # 0,1,2 -> Source names
                .drop(columns="src_idx")
                .rename(columns={"metric": "column"})
        )
        
        sec_aovT = aov[["Source","stars"]].iloc[[1]].rename(columns={"Source": "id"})
        sec_phT = post_hoc[["patient","stars"]][4:].rename(columns={"patient": "id"})
        sec_phT = pd.concat([sec_aovT, sec_phT]).reset_index(drop=True).rename(columns={"stars": col})
        
        if i == 0:
            aov_full = aovT[["Category", "column", col]]
            aov_full["Stat Test"] = "RM-ANOVA"
            aov_full = aov_full[["Stat Test","Category", "column", f"{eval_name} {metric_name}"]]
            
            fph_full = phT[["Category", "column", col]]
            fph_full["Stat Test"] = "Post-Hoc"
            fph_full = fph_full[["Stat Test","Category", "column", f"{eval_name} {metric_name}"]]
            
            fph_sec = sec_phT

        else:
            
            aov_full[col] = aovT[col].to_numpy()        
            fph_full[col] = phT[col].to_numpy()        
            fph_sec[col] = sec_phT[col].to_numpy()
        
        # set unseen variations to yellow color
        if eval_name == "Unseen":
            # ax.set_facecolor('yellow')
            ax1.set_facecolor('grey')
            # ax2.set_facecolor('yellow')
        i += 1

full_stats_df = pd.concat([aov_full,fph_full])
full_stats_df.to_csv('figures/all_stats.csv')
fph_sec = fph_sec.set_index('id')
fph_sec.to_csv('figures/secondary_stats.csv')
fig.set_constrained_layout_pads(
    w_pad=0.05,   # padding around axes (inches)
    h_pad=0.05,
    wspace=0.05,  # space between subplot groups (fraction)
    hspace=0.0
)
fig2.set_constrained_layout_pads(
    w_pad=0.05,   # padding around axes (inches)
    h_pad=0.05,
    wspace=0.05,  # space between subplot groups (fraction)
    hspace=0.0
)

# fig.savefig(f'figures/res_dist.svg')
# fig2.savefig(f'figures/qq.svg')
# plt.show()
# plt.close()


        

        
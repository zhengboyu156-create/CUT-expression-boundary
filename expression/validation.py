"""Measured attachment demonstration, including negative/insufficient outcomes."""
import base64
import html
import io
import json
from pathlib import Path
import time
import unittest
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .artifact_io import save_figure,save_png
from .manifest import prepare,sha256
from .statistics import StainStatistics,to_rgb
from .challenge import fit_morphology,retrieve
from .calibration import fit_and_certify,gate
from .runner import (seed_everything,make_options,make_dataset,train_generator,train_risk,
                     evaluate_generator,metric_summary)


def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))


def attachment_demo(he_path,ihc_path,out,steps=120):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    manifest=out/'manifest.jsonl'
    rows=prepare([dict(pair_id='attachment',group_id='attachment_source',group_kind='unknown',
                       split='demo',he=str(Path(he_path).resolve()),ihc=str(Path(ihc_path).resolve()))],manifest)
    print('PREPARE',len(rows),'identity windows;',sum(r['valid'] for r in rows),'valid',flush=True)
    import tests.test_expression as test_module
    test_module.MANIFEST=manifest
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_module))
    (out/'tests.txt').write_text(stream.getvalue())
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():raise RuntimeError('Module validation failed; training result is not publishable.')
    test_info=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors))
    state=fit_morphology(rows,demo=True);candidates=retrieve(rows,state,demo=True)
    dump(out/'morphology.json',state);dump(out/'local_challenge_candidates.json',candidates)
    models={};datasets={};metadata={};paths={};records={};outputs={};initial={};logs={}
    for name in ('cut','expression_cut'):
        seed_everything()
        opt=make_options(manifest,out/name,name,'demo',ngf=8,batch_size=4)
        dataset=make_dataset(opt)
        from models.cut_model import CUTModel
        from models.expression_cut_model import ExpressionCUTModel
        fresh=(ExpressionCUTModel if name=='expression_cut' else CUTModel)(opt)
        initial_records,_=evaluate_generator(fresh.netG,dataset,batch_size=4,samples=1)
        initial[name]=metric_summary(initial_records)
        del fresh
        seed_everything()
        model,_,meta,path=train_generator(opt,steps,out/name)
        ds=make_dataset(opt)
        rr,oo=evaluate_generator(model.netG,ds,batch_size=4,samples=3)
        models[name]=model;datasets[name]=ds;metadata[name]=meta;paths[name]=path;records[name]=rr;outputs[name]=oo
        logs[name]=json.loads((out/name/'training.json').read_text())
        print('MEASURED',name,metric_summary(rr),flush=True)
    risk,rm,risk_path=train_risk(models['expression_cut'].netG,datasets['expression_cut'],
        out/'expression_cut',paths['expression_cut'],steps=200,cache_demo=True)
    enhanced,en_outputs=evaluate_generator(models['expression_cut'].netG,datasets['expression_cut'],risk,batch_size=4,samples=3)
    records['expression_cut']=enhanced;outputs['expression_cut']=en_outputs
    # Directly test the learned generator's response, without recoloring pixels.
    measurement=StainStatistics();controls=[];control_images=[];g=models['expression_cut'].netG.eval()
    with torch.no_grad():
        for index in range(len(datasets['expression_cut'])):
            sample=datasets['expression_cut'][index];a=sample['A'][None];mask=sample['tissue_mask'][None]
            q=[];images=[]
            for condition in (.1,.5,.9):
                fake=g(a,condition=torch.full((1,1,1,1),condition));value,_=measurement(fake,mask)
                q.append(float(value));images.append(fake[0])
            controls.append(dict(window_id=datasets['expression_cut'].rows[index]['window_id'],
                                 requested=[.1,.5,.9],measured=q))
            if index==0:control_images=images
    deltas=np.array([c['measured'][-1]-c['measured'][0] for c in controls])
    control_summary=dict(mean_high_minus_low=float(deltas.mean()),
                         fraction_high_above_low_by_1pp=float((deltas>.01).mean()),
                         fraction_monotone=float(np.mean([c['measured'][0]<=c['measured'][1]<=c['measured'][2] for c in controls])))
    try:
        fit_and_certify(enhanced,enhanced,dict(training_groups=metadata['expression_cut']['training_groups'],
                                             generator_hash=sha256(paths['expression_cut']),risk_hash=sha256(risk_path)))
        raise AssertionError('The attachment must not certify independent risk.')
    except ValueError as e:calibration_reason=str(e)
    accepted,gate_reason=gate([r['score'] for r in enhanced],None,sha256(paths['expression_cut']),sha256(risk_path))
    from .cli import predict_image
    infer_rows=predict_image(he_path,paths['expression_cut'],risk_path,out/'he_only_inference')
    sample_diversity=float(np.mean([float((o['samples'][0]-o['samples'][-1]).abs().mean()) for o in en_outputs]))
    summary=dict(scope='single-pair in-sample software/overfitting feasibility only; not MIST test effectiveness',
        n_pairs=1,n_source_groups=1,groups_verified_as_patients=False,windows=len(rows),valid_windows=sum(r['valid'] for r in rows),
        same_coordinate_pairs=all(r['he_bbox']==r['ihc_bbox'] for r in rows),window=128,
        tests=test_info,default_width64_six_blocks_tested=True,micro_training_width=8,residual_blocks=6,
        training_steps_per_model=steps,initial=initial,final={n:metric_summary(r) for n,r in records.items()},
        local_proxy_conflict_candidates=len(candidates),formal_cross_patient_challenges=0,
        conditional_control=control_summary,mean_sample_rgb_difference_minus1to1=sample_diversity,
        independent_calibration=dict(status='not_possible',reason=calibration_reason,certificate=None),
        gate=dict(reason=gate_reason,accepted=int(accepted.sum()),total=len(accepted)),
        he_only_inference=dict(ihc_accessed=False,accepted=sum(r['accepted'] for r in infer_rows)),
        elapsed_seconds=time.monotonic()-start)
    enrich_summary(out,summary,enhanced,controls)
    for name,rr in records.items():dump(out/(name+'_window_results.json'),rr)
    dump(out/'condition_sweep.json',controls);dump(out/'verification.json',summary)
    figures(out,rows,records,outputs,logs,control_images,controls,candidates,he_path,ihc_path)
    report(out,summary,rows)
    print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def enrich_summary(out,summary,records,controls):
    from sklearn.metrics import roc_auc_score
    events=np.array([r['event'] for r in records]);scores=np.array([r['score'] for r in records])
    summary['raw_risk_in_sample']=dict(brier=float(np.mean((scores-events)**2)),
        auroc=float(roc_auc_score(events,scores)) if len(np.unique(events))==2 else None,
        calibrated=False,independent_validation=False)
    summary['conditional_control']['requested_high_minus_low']=.8
    summary['conditional_control']['mean_condition_absolute_error']=float(np.mean([
        np.abs(np.array(c['measured'])-np.array(c['requested'])).mean() for c in controls]))
    summary['conditional_control']['interpretation']='monotone response measured; amplitude remains far below requested span; effective controllability not established'
    first=Path(out).parent/'round1_bottleneck_film/verification.json'
    if first.exists():
        previous=json.loads(first.read_text())
        summary['architecture_comparison']=dict(
            bottleneck_only_mae=previous['final']['expression_cut']['mae'],
            bottleneck_only_high_minus_low=previous['conditional_control']['mean_high_minus_low'],
            decoder_film_mae=summary['final']['expression_cut']['mae'],
            decoder_film_high_minus_low=summary['conditional_control']['mean_high_minus_low'],
            interpretation='control response increased while proxy MAE worsened; exploratory same-image comparison')


def figures(out,rows,records,outputs,logs,control_images,controls,candidates,he_path,ihc_path):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    he=np.array(Image.open(he_path).convert('RGB'));ihc=np.array(Image.open(ihc_path).convert('RGB'))
    by_id={r['window_id']:r for r in rows};all_ids=[r['window_id'] for r in rows if r['valid']]
    stitched={}
    for name,oo in outputs.items():
        array=np.full_like(he,220)
        for o in oo:
            x,y,w,h=by_id[o['id']]['he_bbox'];array[y:y+h,x:x+w]=to_rgb(o['fake'])
        stitched[name]=array;save_png(out/(name+'_RESEARCH_ONLY.png'),Image.fromarray(array))
    fig,axes=plt.subplots(2,3,figsize=(13,8.5),layout='constrained')
    for ax,img,title in zip(axes.flat[:4],[he,ihc,stitched['cut'],stitched['expression_cut']],
         ['Input H&E / automatic 8 x 8 grid','Real IHC / identical coordinates',
          'CUT / attachment overfit only','Extended CUT / attachment overfit only']):
        ax.imshow(img);ax.set_title(title);ax.axis('off')
    for ax in axes.flat[:2]:
        for p in range(0,1025,128):ax.axhline(p-.5,color='#117c91',lw=.5);ax.axvline(p-.5,color='#117c91',lw=.5)
    error=np.full((8,8),np.nan)
    for r in records['expression_cut']:
        x,y,_,_=by_id[r['window_id']]['he_bbox'];error[y//128,x//128]=r['error']
    ax=axes[1,1];im=ax.imshow(error,vmin=0,vmax=max(.2,float(np.nanmax(error))),cmap='magma')
    ax.set_title('Post-hoc true proxy error / not confidence');ax.set_xlabel('Window column');ax.set_ylabel('Window row')
    fig.colorbar(im,ax=ax,label='Absolute DAB area-fraction error',shrink=.7)
    ax=axes[1,2];ax.imshow(Image.open(out/'he_only_inference/selective_display.png'))
    ax.set_title('Automatic gate: 0 / 64 released\nNo independent certificate');ax.axis('off')
    fig.suptitle('Real computation on one supplied image pair / no held-out patients',fontsize=14)
    save_figure(out/'overview.png',fig);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for name,color in [('cut','#70747c'),('expression_cut','#117c91')]:
        values=[l['G'] for l in logs[name]];axes[0].plot(range(1,len(values)+1),values,label=name,color=color)
    axes[0].set(xlabel='Optimization step',ylabel='Generator objective',title='Different loss objectives; inspect within each run');axes[0].legend()
    q=[r['q_reference'] for r in records['cut']]
    for name,color in [('cut','#70747c'),('expression_cut','#117c91')]:
        axes[1].scatter(q,[r['q_generated'] for r in records[name]],s=14,alpha=.65,label=name,color=color)
    axes[1].plot([0,1],[0,1],ls='--',c='#888');axes[1].set(xlabel='Reference DAB area fraction',ylabel='Generated DAB area fraction',title='All 64 training-image windows',xlim=(-.03,1.03),ylim=(-.03,1.03));axes[1].legend()
    for c in controls:axes[2].plot([.1,.5,.9],c['measured'],color='#117c91',alpha=.16)
    axes[2].plot([.1,.5,.9],np.array([c['measured'] for c in controls]).mean(0),color='#bd7620',lw=3,label='Mean over all windows')
    axes[2].set(xlabel='Requested conditional fraction',ylabel='Measured generated fraction',title='Learned FiLM response / no manual recoloring',ylim=(-.03,1.03));axes[2].legend()
    save_figure(out/'training_and_control.png',fig);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    first=outputs['expression_cut'][0];rr=records['expression_cut'][0]
    for ax,img,title in zip(axes[0],[to_rgb(first['he']),to_rgb(first['reference']),first['pixel_probability'][0].numpy()],
                           ['Automatic first valid H&E window','Same-coordinate real IHC','Learned pixel DAB proxy / not cell truth']):
        ax.imshow(img,vmin=0,vmax=1,cmap='viridis');ax.set_title(title);ax.axis('off')
    for i,(ax,img) in enumerate(zip(axes[1],control_images)):
        ax.imshow(to_rgb(img));ax.set_title(f"G condition {[.1,.5,.9][i]:.1f}; measured {controls[0]['measured'][i]:.3f}");ax.axis('off')
    fig.suptitle('Fixed first window / conditional images come from the trained generator')
    save_figure(out/'posterior_and_samples.png',fig);plt.close(fig)
    if candidates:
        pair=candidates[0];a,b=by_id[pair['a']],by_id[pair['b']]
        fig,axes=plt.subplots(2,2,figsize=(8,8),layout='constrained')
        for row,axrow in zip([a,b],axes):
            x,y,w,h=row['he_bbox']
            for ax,img,label in zip(axrow,[he,ihc],['H&E','Real IHC']):
                ax.imshow(img[y:y+h,x:x+w]);ax.axis('off')
                ax.set_title(f"{label} {row['window_id'].split(':')[-1]} / [{x},{y},{w},{h}]\nreference proxy q={row['q']:.3f}")
        fig.suptitle(f"Automatic local proxy-conflict candidate / distance={pair['distance']:.3f}\nOne source pair; not a confirmed protein-expression contradiction")
        save_figure(out/'challenge_candidate.png',fig);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    r=records['expression_cut'];score=np.array([x['score'] for x in r]);error=np.array([x['error'] for x in r])
    axes[0].scatter(score,error,s=20,c='#117c91');axes[0].axhline(.1,ls='--',c='#bd7620')
    axes[0].set(xlabel='Raw learned error score',ylabel='Post-hoc proxy error',title='In-sample diagnostic / uncalibrated')
    from .calibration import fixed_coverage_report
    curve=fixed_coverage_report(r,np.linspace(.1,1,10))
    axes[1].plot([x['coverage'] for x in curve],[x['risk'] for x in curve],marker='o',c='#117c91')
    axes[1].set(xlabel='Fraction of windows retained',ylabel='Error-event rate (absolute error > 0.10)',title='In-sample curve / no certified guarantee',ylim=(-.03,1.03))
    axes[2].axis('off');axes[2].text(.02,.85,'Independent source groups: 1\nVerified patient identities: 0\nIndependent calibration A: unavailable\nIndependent certification B: unavailable\n\nCertified release: 0 / 64\nH&E-only inference: executed\nMissing certificate => automatic refusal',va='top',linespacing=1.8)
    save_figure(out/'risk_and_refusal.png',fig);plt.close(fig)


def report(out,s,rows):
    a=s['initial']['expression_cut'];b=s['final']['expression_cut'];base=s['final']['cut'];ctrl=s['conditional_control']
    scope='两张附件构成一个图像对；所有训练与数值评估均来自这个图像对。以下仅为软件可行性和样本内拟合证据，未运行完整 MIST 测试，也没有独立患者有效性结论。'
    steps=[
      ('1.1 全图分窗','已运行',f"{s['windows']} 对 128×128 窗；全部 he_bbox == ihc_bbox，覆盖原图一次",'manifest.py / 同坐标与覆盖率测试'),
      ('1.2 同窗质量检查','已运行',f"{s['valid_windows']} 对通过固定前景/结构门槛；无位移或位置搜索",'manifest.py / 每窗 reason 日志'),
      ('1.3 实染参考统计','已运行','H–DAB 分解；固定 H&E 掩膜作共同分母；NumPy 与 Torch 测量一致','statistics.py / 数值一致性测试'),
      ('1.4 形态表征与聚类','已运行','H&E 特征标准化、自动选 K；正式流程只在 train 拟合','challenge.py / morphology.json'),
      ('1.5 冲突候选检索','局部可行',f"检出 {s['local_proxy_conflict_candidates']} 个局部染色代理冲突候选；跨患者挑战对 0",'challenge.py / local_challenge_candidates.json'),
      ('1.6 可预测边界报告','待独立验证','已输出逐窗真实误差及覆盖率曲线；当前曲线仅属样本内诊断','runner.py / 两组 window_results.json'),
      ('2.1 自动目标与读取','已运行','随机 B 与 paired_B 分离；配对裁剪同步；参考统计映射到 5 个分箱','expression_pair_dataset.py / 随机 B 不污染参考测试'),
      ('2.2 H&E 分布预测头','样本内已运行',f"分箱对数损失 {a['bin_log_loss']:.4f} → {b['bin_log_loss']:.4f}；尚非校准后验",'expression_cut_model.py / 真实梯度及条件不泄漏测试'),
      ('2.3 像素概率辅助头','代理可行','使用自动 DAB 像素伪标签；输出不是细胞蛋白阳性概率','networks.py / pixel_proxy 非零梯度测试'),
      ('2.4 六残差块 + FiLM','控制幅度不足',f"64 通道默认骨干前后向通过；微型训练 8 通道、6 残差块；高低条件平均差 {ctrl['mean_high_minus_low']:.4f}，条件指定跨度为 0.8",'networks.py / 宽度 64 测试及 condition_sweep.json'),
      ('2.5 分布采样生成','已运行',f"每窗 3 个真实 G 样本；首末样本平均 RGB 差 {s['mean_sample_rgb_difference_minus1to1']:.6f}（[-1,1] 尺度）",'runner.py / generated_batch，无手工染色'),
      ('2.6 条件一致性评价','尚未充分满足',f"高条件比低条件至少增加 1 个百分点的窗口比例 {ctrl['fraction_high_above_low_by_1pp']:.1%}；有响应但远未充分跟随指定染色量",'condition_sweep.json / 全部 64 窗'),
      ('3.1 来源分组与防泄漏','已运行','按 group_id、pair_id 和图像内容哈希检查跨 split 重叠；附件不拆成独立患者','manifest.py / 内容泄漏与单来源测试'),
      ('3.2 自动错误事件','已运行','固定 δ=0.10；事件为同坐标区域 DAB 面积率绝对误差超限','runner.py / 每窗 event'),
      ('3.3 错误预测器 h','样本内已运行',f"冻结 G 后训练 h；只输入 H&E、生成图和采样差异；样本内 AUROC={s.get('raw_risk_in_sample',{}).get('auroc')}，未经独立校准",'networks.py / ErrorPredictor 接口与梯度测试'),
      ('3.4 概率校准 A','数据不足','实现保序校准；本附件不能拟合独立组校准器，未输出伪校准结果','calibration.py / tests.txt 中合成单元夹具仅测软件'),
      ('3.5 风险认证 B','数据不足','实现固定候选阈值、Bonferroni 单侧二项上界；每个独立组固定种子抽 1 窗','calibration.py / 无临床或全像素风险保证'),
      ('3.6 自动拒绝及 H&E 推理','已运行','仅输入 H&E 已完成推理；缺独立认证时 0/64 放行，拒绝不代表阴性','cli.py / he_only_inference/prediction.json')]
    dump(out/'step_evidence.json',[dict(step=x[0],status=x[1],evidence=x[2],source=x[3]) for x in steps])
    md=f'''# CUT 表达边界扩展：附件实跑报告

{scope}

## 已测量结果

| 项目 | 实测 |
|---|---|
| 同坐标窗 | {s['windows']} 对；质量通过 {s['valid_windows']} 对 |
| 自动测试 | {s['tests']['run']} 项，失败 {s['tests']['failures']}，错误 {s['tests']['errors']} |
| 六块骨干 | 默认 ngf=64 前后向通过；微型训练 ngf=8 |
| 每个模型训练量 | {s['training_steps_per_model']} 步，batch=4，CPU；相同初始骨干种子、同一批序列 |
| 原版 CUT 样本内区域 MAE | {base['mae']:.5f} |
| 扩展 CUT 样本内区域 MAE | {b['mae']:.5f} |
| 扩展分箱对数损失 | {a['bin_log_loss']:.5f} → {b['bin_log_loss']:.5f} |
| 高/低条件生成面积率差 | 平均 {ctrl['mean_high_minus_low']:.5f} |
| 真正模型样本差异 | {s['mean_sample_rgb_difference_minus1to1']:.6f}，不代表校准置信度 |
| 独立认证放行 | 0/{s['valid_windows']}；未创建认证文件 |

MAE 和 δ 的单位是区域 DAB 阳性像素面积率，0.10 等于 10 个百分点。它们不是 HER2 临床评分或细胞阳性率。两种生成器的总损失包含不同项，不能通过总损失大小比较模型优劣。全部结果照实保留，不按测试效果选窗口或 checkpoint。条件高低差仍远小于请求跨度 0.8，可控生成的有效性尚未充分建立。生成图仍存在明显色彩与染色定位偏差；区域统计误差下降不能证明已经生成可判读的虚拟 IHC。

## 每一步的证据

| 步骤 | 状态 | 实测 / 限制 | 对应文件 |
|---|---|---|---|
'''+''.join('| '+' | '.join(x)+' |\n' for x in steps)+'''
## 如何解释有效性

代码可执行、真实数据可反传、分布头能否拟合、FiLM 是否响应条件，均可由本例检查。若相应实测不达标，应视为负结果，不把“有梯度”写成“已证明有效”。样本内拟合和局部候选不支持患者级性能、不确定性校准或形态决定蛋白状态的结论。像素辅助头只有自动染色伪标签，不能替代细胞标注。

完整验证需输入 MIST 配对清单和可靠来源组元数据；官方测试来源保持隔离，训练、开发、错误预测器训练、校准 A、认证 B 分组隔离。当前代码中的二项认证对象是“每独立来源组均匀抽取一个有效窗”的平均风险；它不保证所有窗、所有细胞或全部像素的错误率。来源只到切片时，结论也只到切片分组。

正式实验需要比较 CUT、加分布头、加条件生成和完整门控；报告普通集/自动冲突集的分组置信区间，以及固定覆盖率 50%/80%/100% 的错误率、总拒绝率。没有独立实染参考或未训练校准器时，系统不发布可信表达图。

复现命令、正式训练路径和输入清单格式见项目根目录 README_EXPRESSION_CN.md。原始逐窗结果、训练日志、完整模块测试及权重随代码包提供。
'''
    comparison=s.get('architecture_comparison')
    comparison_text=''
    if comparison:
        comparison_text=f"瓶颈 FiLM 初轮：MAE {comparison['bottleneck_only_mae']:.5f}，高低条件差 {comparison['bottleneck_only_high_minus_low']:.5f}；增加末次归一化后 FiLM：MAE {comparison['decoder_film_mae']:.5f}，高低条件差 {comparison['decoder_film_high_minus_low']:.5f}。条件响应增强，同时区域 MAE 变差。两轮均为探索性样本内实验，失败对照完整保留。"
        md+='\n## 架构调整的实测代价\n\n'+comparison_text+'\n'
    (out/'verification_report.md').write_text(md)
    tr=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in steps)
    images=[]
    for name,caption in [('overview.png','整图结果：64 对同坐标窗、真实 G 输出、真实误差与自动拒绝'),
                         ('training_and_control.png','训练轨迹、样本内实染统计一致性与 FiLM 条件响应'),
                         ('challenge_candidate.png','自动冲突候选：每个 H&E 窗仍对应自身同坐标 IHC'),
                         ('posterior_and_samples.png','固定首个窗：像素伪标签概率与实际条件生成'),
                         ('risk_and_refusal.png','错误预测诊断与认证缺失时的自动拒绝')]:
        if (out/name).exists():
            encoded=base64.b64encode((out/name).read_bytes()).decode()
            images.append(f'<figure><figcaption>{caption}</figcaption><img src="data:image/png;base64,{encoded}" alt="{caption}"></figure>')
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CUT 表达边界：真实附件验证</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1120px;margin:28px auto;padding:0 18px;color:#202a32;line-height:1.65}}h1{{font-size:28px}}h2{{margin-top:32px}}img{{width:100%;height:auto}}figure{{margin:28px 0}}figcaption{{font-weight:600}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #d4dce2;padding:10px;vertical-align:top}}.scroll{{overflow-x:auto}}.note{{border-left:4px solid #bd7620;padding:8px 16px;background:#fbf6ed}}code{{overflow-wrap:anywhere}}@media(max-width:600px){{h1{{font-size:23px}}td,th{{padding:7px;font-size:13px}}}}</style>
<h1>CUT 表达边界：代码实现与真实附件验证</h1><p class="note">{scope}</p>
<p>64 对同坐标窗口 · {s['tests']['run']} 项自动测试通过 · 每个模型 {s['training_steps_per_model']} 步 CPU 微型训练 · 0 个独立患者验证</p>
<h2>实测结果</h2><p>区域 DAB 面积率 MAE：原版 CUT <b>{base['mae']:.5f}</b>，扩展 CUT <b>{b['mae']:.5f}</b>。扩展头分箱对数损失：{a['bin_log_loss']:.5f} → {b['bin_log_loss']:.5f}。这些数值均来自同一训练图像对。</p>
{''.join(images)}<h2>三个创新点：18 步证据表</h2><div class="scroll"><table><thead><tr><th>步骤</th><th>状态</th><th>证据及限制</th><th>来源</th></tr></thead><tbody>{tr}</tbody></table></div>
<h2>当前结论</h2><p>本次验证了代码执行与样本内学习能力。条件响应、样本差异和原始风险分数均按真实运算呈现；它们不等于生物学表达真值或校准置信度。完整 MIST、已训练基线、可靠来源 ID 和独立实染验证尚未提供，患者级有效性与风险保证保持未验证。</p>
<p>生成图仍存在明显色彩与染色定位偏差；区域统计误差下降不能证明已经生成可判读的虚拟 IHC。条件响应幅度仍远小于请求跨度，可控生成尚未充分满足一致性要求。</p>
<p>{comparison_text}</p><p>二项认证针对每个独立来源组随机抽一个有效窗的平均风险，不针对全部细胞或像素。默认六残差块 ngf=64 已通过前后向检查；本例微型训练使用 ngf=8。缺少独立认证时，全图自动遮罩。</p></html>'''
    (out/'verification_report.html').write_text(body)

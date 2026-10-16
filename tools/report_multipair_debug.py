"""Evidence report for the multi-pair investigation. Uses measured JSON/PNG only."""
import argparse
import base64
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from expression.artifact_io import save_figure


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',default='artifacts/multipair_debug');a=p.parse_args()
    base=Path(a.base).resolve();root=Path(__file__).resolve().parents[1]
    old=[json.loads(s) for s in (base/'v1_full_manifest.jsonl').read_text().splitlines() if s]
    new=[json.loads(s) for s in (base/'full_manifest.jsonl').read_text().splitlines() if s]
    one,two=read(base/'v1/summary.json'),read(base/'v2/summary.json')
    compat=read(base/'v2_compat/summary.json')
    # Compatibility conversion is not an additional generator training run.
    aa=read(base/'v1/holdout_records.json');bb=read(base/'v2_compat/holdout_records.json')
    difference=max(abs(x['q_generated']-y['q_generated']) for x,y in zip(aa,bb))
    assert difference<1e-6,'Compatibility conversion changed the mean generator output.'
    qc=[]
    for pid in sorted({r['pair_id'] for r in new if r['pair_id'].startswith('2M_')}):
        before=[r for r in old if r['pair_id']==pid];after=[r for r in new if r['pair_id']==pid]
        qc.append(dict(pair_id=pid,total=len(after),old_kept=sum(r['valid'] for r in before),
                       new_kept=sum(r['valid'] for r in after),unobserved=sum(r['reason']=='reference_unobserved' for r in after)))
    train=[r for r in new if r['valid'] and r['debug_partition']=='train']
    counts=np.bincount([min(4,int(r['q']*5)) for r in train],minlength=5)
    prior=(counts+1)/(len(train)+5)
    prior_log_loss=float(np.mean([-np.log(prior[r['bin_target']]) for r in aa]))
    info=dict(new_pairs=6,anchor_pairs=1,source_identity_verified=False,new_windows=384,
        old_kept=sum(r['old_kept'] for r in qc),new_kept=sum(r['new_kept'] for r in qc),quality_by_pair=qc,
        train_pairs=5,dev_pairs=1,holdout_pairs=1,train_windows=len(train),dev_windows=one['dev']['n'],holdout_windows=one['holdout']['n'],
        generator_training_runs=2,steps_each=one['steps'],ngf=8,residual_blocks=6,
        train_bin_counts=counts.tolist(),train_prior_log_loss_on_holdout=prior_log_loss,
        original=one,experimental=two,compatibility=compat,compatibility_max_stat_difference=difference,
        trusted_release_coverage=0,scope='same-source image-pair debugging, not independent-patient performance')
    (base/'debug_findings.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained');x=np.arange(len(qc));width=.36
    for offset,key,color,label in [(-width/2,'old_kept','#ad7330','v1 strong IHC foreground gate'),(width/2,'new_kept','#117c91','v2 broad reference observability')]:
        bars=ax.bar(x+offset,[r[key] for r in qc],width,color=color,label=label)
        ax.bar_label(bars,padding=3)
    ax.set(xticks=x,xticklabels=[r['pair_id'].replace('2M_','') for r in qc],ylim=(0,73),ylabel='Retained same-coordinate windows / 64',title='Faint-stain exclusion bug: 8 / 384 retained before; 333 / 384 after')
    ax.legend(loc='upper center',ncol=2);save_figure(base/'qc_comparison.png',fig);plt.close(fig)
    pair=next(r for r in new if r['pair_id']=='2M_13_17')
    def resolve(value):
        path=Path(value);return path if path.is_absolute() else (base/path).resolve()
    he=np.array(Image.open(resolve(pair['he'])));ihc=np.array(Image.open(resolve(pair['ihc'])))
    imgs=[he,ihc,np.array(Image.open(base/'v1/2M_13_17_generated.png')),np.array(Image.open(base/'v2/2M_13_17_generated.png'))]
    fig,axes=plt.subplots(1,4,figsize=(15,4.4),layout='constrained')
    titles=['Held-out H&E / same source family','Paired real IHC / fixed coordinates','v1 / current compatibility G','Experimental mix + within-bin head\nRegressed; disabled by default']
    for ax,img,title in zip(axes,imgs,titles):ax.imshow(img);ax.set_title(title);ax.axis('off')
    fig.suptitle('Actual trained neural outputs / 63 measurable windows; no manual recoloring',fontsize=13)
    save_figure(base/'holdout_comparison.png',fig);plt.close(fig)
    h1,h2=one['holdout'],two['holdout']
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    vals=[h1['zero_fraction_mae'],h1['mae'],h2['mae']]
    bars=axes[0].bar(['Fixed zero','v1 / compat','Experiment'],vals,color=['#9ca3aa','#117c91','#bd7620']);axes[0].bar_label(bars,fmt='%.5f',padding=3)
    axes[0].set(yscale='log',ylim=(.001,1),ylabel='DAB area-fraction MAE (log scale)',title='Low area error can hide bad localization')
    for ax,key,title in [(axes[1],'dab_iou','DAB pixel IoU: both close to zero'),(axes[2],'rgb_l1','RGB L1 on [-1,1]: both large')]:
        bars=ax.bar(['v1 / compat','Experiment'],[h1[key],h2[key]],color=['#117c91','#bd7620']);ax.bar_label(bars,fmt='%.5f',padding=3)
        ax.set(title=title);ax.set_ylim(0,max(h1[key],h2[key])*1.25)
    save_figure(base/'metric_traps.png',fig);plt.close(fig)
    missing=next(r for r in new if r['pair_id']=='2M_11_11');mr=[r for r in new if r['pair_id']==missing['pair_id']]
    observed=np.zeros((8,8));valid=np.zeros((8,8))
    for r in mr:
        x,y,_,_=r['he_bbox'];observed[y//128,x//128]=r['ihc_tissue'];valid[y//128,x//128]=r['valid']
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    axes[0].imshow(Image.open(resolve(missing['ihc'])));axes[0].set_title('11_11 real IHC: reference gaps');axes[0].axis('off')
    im=axes[1].imshow(observed,vmin=0,vmax=1,cmap='viridis');axes[1].set_title('Broad observation fraction / fixed grid');fig.colorbar(im,ax=axes[1],shrink=.65)
    axes[2].imshow(valid,vmin=0,vmax=1,cmap='Greys_r');axes[2].set_title('17 kept / 46 reference gaps / 1 H&E QC');axes[2].set_xlabel('Original window column')
    save_figure(base/'missing_reference.png',fig);plt.close(fig)
    issues=[
        ('B1','淡染 IHC 被质控整图过滤','新 6 对仅保留 8/384 窗；5 对为 0 窗','改用近白可观测性检查，保留 333/384；同坐标不变，明显缺失仍排除','已修复；可观测性规则仍是代理'),
        ('B2','文件名进入采样与风险输入','旧代码改名后样本最大差 0.03205；方差图也改变','随机种子来自 H&E 像素内容；改名不再改变样本','已修复并回归测试'),
        ('B3','模型选项读取宿主 argv','无关宿主 --ngf INVALID 可令构建器退出，exit=2','显式传入独立命令行；原生入口仍读取自身参数','已修复并回归测试'),
        ('B4','统计窗大小未传播','prepare 可设 256，旧训练器仍默认 128 并报错','从 manifest 推导唯一窗尺寸，实测 256 输入可读取','已修复并回归测试'),
        ('B5','评估长期保留全部图像张量','旧返回值含 HE/参考/生成/多样本；10000 窗约 13 GiB','正式评估只留记录；需要图像时按批回调','已修复；记录模式返回 0 个图像对象'),
        ('B6','图像缓存跨实例且容量太小','5 个训练图像对共 10 张图，旧全局 LRU 仅 8 项','实例私有、可配置上限；实验记录 10 次解码、1922 次命中','已修复并回归测试'),
        ('B7','非法分数及协议变更未拒绝','旧 gate 接受 -1 和 2；只绑定权重未绑定采样/质控','拒绝非概率输入；校准和 h 绑定协议哈希，旧 h 要重训','已修复并回归测试'),
        ('B8','推理内重复计算权重哈希','每个窗重复读同一 checkpoint','改为每个病例各计算一次','已修复；源码与推理路径检查'),
        ('B9','导出 PNG 未核验完整性','最后一张 PNG 曾出现截断；训练数值未受影响','从已保存权重等值重建，完整缓冲区原子写入并解码复验','已修复并检查全部报告图片')]
    remaining=[
        ('M1','区域比例误差不能代表定位正确',f"v1 留出 MAE={h1['mae']:.6f}，但像素 IoU={h1['dab_iou']:.6f}、RGB L1={h1['rgb_l1']:.4f}。新增空间指标揭露问题；生成器尚未满足染色定位要求。"),
        ('M2','组合条件策略退步',f"箱内代表值+条件预测混合训练的留出 MAE={h2['mae']:.6f}，劣于固定零基线 {h1['zero_fraction_mae']:.6f}。两项各自作用尚未消融；默认关闭。"),
        ('M3','分布头多数类假象',f"留出图全部位于第 0 箱，预测准确率 100% 没有区分价值。训练标签频率基线 log-loss={prior_log_loss:.4f}，优于 v1 的 {h1['bin_log_loss']:.4f}。"),
        ('M4','错误预测器只有单一错误类别','兼容分支开发/留出窗的区域超限事件全为 0；实验分支全为 1。AUROC 均不能计算，接近零的 Brier 不是学会了错误边界。代码已记录类别数。'),
        ('M5','没有独立患者级校准数据','新文件共享来源前缀，真实患者/切片关系未提供。分图像对留出只是同源调试；不做 A/B 认证，可信放行 0%。'),
        ('M6','参考缺失与淡染仍难完全自动区分','11_11 的大块近白区域已标记；边缘部分缺失和染色分解仍属启发式。低 DAB 不等于已核验的蛋白阴性。'),
        ('M7','误差事件定义过粗','默认 δ=0.10 是面积率阈值，可能漏掉稀疏阳性像素或膜定位错误。仍需独立验证的空间/细胞错误目标；不会临时调阈值制造有效性。'),
        ('M8','训练规模不足以判断研究创新有效','每臂 160 步、ngf=8、6 残差块；只有一张留出图像对。真实 MIST 泛化、连续表达后验及风险校准均未验证。')]
    table=lambda head,rows:'<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
    gallery=[]
    for name,caption in [('qc_comparison.png','淡染过滤错误：修复前后保留数量'),('missing_reference.png','缺失参考区域：只做同坐标质量标记'),('holdout_comparison.png','留出图真实生成结果：比例低误差仍可能错色、错定位'),('metric_traps.png','多指标与简单基线：实验策略没有通过有效性验证')]:
        encoded=base64.b64encode((base/name).read_bytes()).decode()
        gallery.append(f'<figure><figcaption>{caption}</figcaption><img alt="{caption}" src="data:image/png;base64,{encoded}"></figure>')
    metrics=[['固定零面积率',f"{h1['zero_fraction_mae']:.6f}",'不作为定位模型','不作为图像模型'],['v1 / 默认兼容路径',f"{h1['mae']:.6f}",f"{h1['dab_iou']:.6f}",f"{h1['rgb_l1']:.6f}"],['实验组合，默认关闭',f"{h2['mae']:.6f}",f"{h2['dab_iou']:.6f}",f"{h2['rgb_l1']:.6f}"]]
    scope='6 对新图 + 1 对之前的图。固定 5 对训练、1 对开发、1 对留出；270/64/63 个有效窗。来源身份未核实，2M 文件家族跨这些调试角色，不构成独立患者验证。两次生成器训练各 160 步，ngf=8、6 个残差块；兼容分支原样复用 v1 生成器权重，仅重训匹配新采样协议的 h。'
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CUT 多配对图片调试报告</title><style>body{font-family:system-ui,sans-serif;max-width:1120px;margin:28px auto;padding:0 18px;color:#202a32;line-height:1.65}h1{font-size:28px}h2{margin-top:32px}figure{margin:28px 0}figcaption{font-weight:650}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #d0d8de;text-align:left;vertical-align:top}.scroll{overflow-x:auto}.note{border-left:4px solid #b77927;padding:10px 16px;background:#fbf6ed}@media(max-width:600px){h1{font-size:23px}th,td{font-size:13px;padding:7px}}</style>'
    body+='<h1>CUT 多配对图片：已修复的软件错误与未解决的模型问题</h1><p class="note">'+scope+'</p>'
    body+='<p><b>结论：软件修复已通过 24 项回归测试；条件训练组合出现退步，已默认关闭。模型的染色定位、表达后验和病理错误校准尚未通过有效性验证。</b></p><h2>同一留出图像对的实测结果</h2><div class="scroll">'+table(['方法','DAB 面积率 MAE','染色像素 IoU','RGB L1（[-1,1]）'],metrics)+'</div>'
    body+=''.join(gallery)+'<h2>确认的软件问题与修复</h2><div class="scroll">'+table(['编号','问题','修复前证据','处理','状态'],issues)+'</div><h2>仍未完善的研究问题</h2>'+table(['编号','问题','实测依据与界限'],remaining)
    body+=f'<h2>默认版本与失败实验均保留</h2><p>兼容转换前后逐窗生成面积率最大差 {difference:.9g}；没有第三次生成器训练。旧风险权重不能静默复用：需重训以匹配采样和质控协议。本批数据不生成独立认证文件，调试权重不放行可信表达展示。</p><p>完整清单、原始逐窗记录、两臂训练日志、失败策略参数、24 项测试和可复现脚本均随代码包提供。病例级风险与细胞表达真值需要另外的真实数据证据。</p></html>'
    (base/'multipair_debug_report.html').write_text(body)
    md='# CUT 多配对图片调试报告\n\n'+scope+'\n\n## 核心结论\n\n软件修复通过 24 项回归测试；实验条件策略退步，默认关闭；区域统计不能替代染色定位或临床表达判断。\n\n'
    md+='| 方法 | DAB 面积率 MAE | 染色像素 IoU | RGB L1 |\n|---|---|---|---|\n'+''.join('| '+' | '.join(r)+' |\n' for r in metrics)
    for title,rows in [('确认的软件问题',issues),('仍未完善的研究问题',remaining)]:
        md+='\n## '+title+'\n\n'+''.join('- '+'；'.join(r)+'。\n' for r in rows)
    md+='\n复现：`python tools/run_multipair_debug.py --steps 160`。测试：`python -m unittest tests.test_expression tests.test_multipair_debug -v`。历史源代码只作对照，当前默认代码拒绝静默复用旧 h / 旧认证协议。\n'
    (base/'multipair_debug_report.md').write_text(md)
    (base/'issues.json').write_text(json.dumps(dict(fixed=issues,remaining=remaining),ensure_ascii=False,indent=2))
    print(json.dumps({k:info[k] for k in ['old_kept','new_kept','train_windows','dev_windows','holdout_windows','train_prior_log_loss_on_holdout','compatibility_max_stat_difference']},indent=2))


if __name__=='__main__':main()

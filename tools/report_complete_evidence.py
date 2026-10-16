"""Render the evidence ledger from measurements; no fabricated positive result."""
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
from expression.manifest import read_manifest,sha256
from expression.artifact_io import save_figure


def read(path):return json.loads(Path(path).read_text())
def table(head,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='artifacts/multipair_debug');p.add_argument('--out',default='artifacts/complete_evidence');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];base=Path(a.base).resolve();out=Path(a.out).resolve()
    d=read(out/'complete_evidence.json');m=d['models']['v2_compat'];ex=d['models']['v2'];c=d['challenge'];cert=d['certification']
    rows=read_manifest(base/'full_manifest.jsonl');by_id={r['window_id']:r for r in rows}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    bars=axes[0].bar(['Old cap','Full radius','Dev + holdout'],[c['old_top16_candidates'],c['complete_candidates'],c['unseen_image_pair_candidates']],color=['#8997a0','#087e8b','#bb5c35']);axes[0].bar_label(bars,padding=3)
    axes[0].set(title='Challenge candidates: code fixed,\nunseen-image evidence absent',ylabel='Proxy-conflict pairs',ylim=(0,125))
    x=np.arange(2);w=.34
    for offset,values,label,color in [(-w/2,[m['bin_log_loss'],m['pixel_bce']],'Trained model','#ba5936'),(w/2,[m['train_histogram_log_loss'],m['pixel_train_prevalence_bce']],'Train-only constant controls','#087e8b')]:
        bars=axes[1].bar(x+offset,values,w,label=label,color=color);axes[1].bar_label(bars,fmt='%.3f',padding=3)
    axes[1].set(xticks=x,xticklabels=['Region-bin log loss','Pixel DAB BCE'],title='Lower is better:\nmodel does not beat simple controls',ylim=(0,1.2));axes[1].legend(fontsize=8)
    axes[1].text(.01,-.2,'Both displayed controls use training targets only.',transform=axes[1].transAxes,fontsize=8)
    for variant,label,color in [('v2_compat','Default / compat','#087e8b'),('v2','Disabled experiment','#ba5936')]:
        sweep=read(out/f'{variant}_condition_sweep.json');x=[r['condition'] for r in sweep[0]['points']];y=np.mean([[r['measured_q'] for r in s['points']] for s in sweep],axis=0)
        axes[2].plot(x,y,'o-',label=label,color=color)
    axes[2].plot([0,1],[0,1],'--',color='#88939b',label='Ideal statistic control')
    axes[2].set(xlabel='Requested DAB fraction',ylabel='Measured generated DAB fraction',title='Actual condition control on all 63 windows',xlim=(0,1),ylim=(0,1));axes[2].legend(fontsize=8)
    save_figure(out/'evidence_summary.png',fig);plt.close(fig)
    s=read(out/'selected_sampling_evidence.json')['v2_compat'];npz=np.load(out/'v2_compat_pixel_evidence.npz')
    fig,axes=plt.subplots(2,3,figsize=(11,7.8),layout='constrained')
    for ax,name,title in zip(axes[0],['he','reference','generated'],['H&E / fixed coordinates','Real paired IHC','Actual trained virtual IHC']):
        ax.imshow(Image.open(out/f'v2_compat_{name}_crop.png'));ax.set_title(title);ax.axis('off')
    axes[1,0].imshow(npz['reference_positive']&npz['mask'],vmin=0,vmax=1,cmap='gray');axes[1,0].set_title('Reference DAB proxy > 0.20')
    axes[1,1].imshow(npz['generated_positive']&npz['mask'],vmin=0,vmax=1,cmap='gray');axes[1,1].set_title('Generated DAB proxy > 0.20')
    im=axes[1,2].imshow(npz['pixel_probability'],vmin=0,vmax=1,cmap='viridis');axes[1,2].set_title('Auxiliary pixel DAB probability');fig.colorbar(im,ax=axes[1,2],shrink=.7)
    for ax in axes[1]:ax.axis('off')
    fig.suptitle(f"Automatically selected maximum-reference-DAB holdout window: {s['window_id']}\nBbox={s['bbox']}; reference fraction={s['reference_q']:.5f}. No manual ROI.",fontsize=11)
    save_figure(out/'actual_window_evidence.png',fig);plt.close(fig)
    candidate=read(out/'complete_challenge_candidates.json')[0]
    fig,axes=plt.subplots(1,4,figsize=(12,4.2),layout='constrained')
    for i,key in enumerate(['a','b']):
        r=by_id[candidate[key]];x,y,w,h=r['he_bbox']
        for j,image_key in enumerate(['he','ihc']):
            crop=np.array(Image.open(r[image_key]))[y:y+h,x:x+w]
            ax=axes[i*2+j];ax.imshow(crop);ax.axis('off');ax.set_title(f"{key.upper()} {image_key.upper()} / q={r['q']:.3f}\n{r['window_id']}\n{x},{y},{w},{h}",fontsize=9)
    fig.suptitle(f"Automatic nearest qualifying H&E proxy pair: distance={candidate['distance']:.3f}; DAB difference={candidate['delta']:.3f}\nContains training/demo images. This is not a verified opposite-protein patient challenge.",fontsize=11)
    save_figure(out/'automatic_candidate_evidence.png',fig);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for variant,label,color in [('v2_compat','Default / compat','#087e8b'),('v2','Disabled experiment','#ba5936')]:
        r=d['models'][variant]['fixed_coverage'];axes[0].plot([x['coverage'] for x in r],[x['area_error_rate'] for x in r],'o-',label=label,color=color)
    axes[0].plot([.5,.8,1.],[1,1,1],'x--',color='#38434b',label='Default localization diagnostic')
    axes[0].set(xlabel='Requested within-image coverage',ylabel='Observed failure proportion',title='Raw score ranking: single-class area events',ylim=(-.05,1.08));axes[0].legend(fontsize=8)
    ns=[1,2,10,46,47,100];upper=[1-(.05/7)**(1/n) for n in ns]
    axes[1].plot(ns,upper,'o-',color='#087e8b');axes[1].axhline(.1,color='#ba5936',linestyle='--',label='Risk budget 0.10')
    axes[1].set(xlabel='Accepted independent B groups, zero errors',ylabel='One-sided upper risk bound',title='Mathematical software check / K=7, alpha=.05',ylim=(0,1));axes[1].legend(fontsize=8)
    fig.suptitle('No actual independent-patient certificate exists. Rejection is not a negative protein label.',fontsize=11)
    save_figure(out/'risk_and_certification_evidence.png',fig);plt.close(fig)
    steps=[
      ('1.1','自动全图分窗','软件实测通过','7 对、448 窗；每张图覆盖一次；全部 H&E/IHC 框相同。这只验证坐标实现，不证明实际配准精度。','input_identity.json'),
      ('1.2','自动同坐标质控','规则执行通过，组织真值未验证','新 6 对 8→333/384 窗；仍是近白可观测性规则，不证明病理组织判断正确。','input_identity.json；旧/新 manifest'),
      ('1.3','实染参考量测','数值复算通过',f"独立 NumPy 与 manifest 的 q 最大差 {d['max_independent_reference_difference']:.3g}；q 只是区域 DAB 代理。",'independent_reference_measurements.json'),
      ('1.4','形态表示训练','软件实测通过','仅用训练角色 270 窗拟合 9 维 H&E 特征及聚类；固定距离 1.5、差异 0.2。','morphology_train_only.json'),
      ('1.5','形态近似、表达相反挑战','研究证据不足','全量含训练图得到 106 对代理候选；开发/留出 127 窗得到 0 对；无经核实的蛋白相反标签。','challenge_audit.json；完整候选清单'),
      ('1.6','标记物特异可预测边界','未证明','标记物身份、病人关系和真实表达标签未核实，当前不能宣称得到 HER2 等标记物的可信边界。','complete_evidence.json'),
      ('2.1','配对监督与目标自动派生','软件实测通过','随机 B 与 paired_B 分离；配对同步；五箱标签为固定 DAB 统计。','tests.txt；逐窗独立记录'),
      ('2.2','区域表达分布头','当前实测不支持有效性',f"留出 log-loss={m['bin_log_loss']:.4f}，劣于训练直方图先验 {m['train_histogram_log_loss']:.4f}；全部 63 窗属于第 0 箱。",'v2_compat_independent_holdout_records.json'),
      ('2.3','细胞阳性概率／像素头','细胞层面未实现；像素代理表现弱',f"实现的是像素 DAB 辅助头，无细胞检测与细胞真值。像素 BCE={m['pixel_bce']:.4f}；训练阳性率常数基线 BCE={m['pixel_train_prevalence_bce']:.4f}。",'v2_compat_pixel_evidence.npz；complete_evidence.json'),
      ('2.4','六残差块与 FiLM 条件控制','结构通过，控制量不足',f"高低条件跨度 0.8，生成 DAB 面积率均差仅 {m['high_low_q_response']:.6f}；不能把单调响应当作定量控制成功。",'v2_compat_condition_sweep.json'),
      ('2.5','多样本后验生成','采样可运行，生物学后验未证明','真实 G 生成样本与采样方差已保存；单参考、分箱代表值和未校准概率均不足以识别真实生物学后验。','selected_sampling_evidence.json；sample PNG'),
      ('2.6','条件／空间一致性','当前实测失败',f"默认 MAE={m['area_mae']:.6f} 但 IoU={m['mean_dab_iou']:.6f}；实验组合 MAE={ex['area_mae']:.6f}，保持默认关闭。",'独立逐窗 CSV；condition_sweep.json'),
      ('3.1','来源分组与防泄漏','拒绝逻辑通过，数据不独立','仅 2 个未核实来源家族；2M 家族跨调试角色；开发与留出源重叠，不能改名伪造独立组。','certification_audit.json；tests.txt'),
      ('3.2','病理错误事件定义','面积率事件可算，病理错误目标不足','默认 63 窗面积率事件全 0，但 IoU<0.1 的定位诊断全失败。诊断不是临床金标准，也没有事后替换训练目标。','独立逐窗 CSV'),
      ('3.3','h 预测真实错误','未证明','默认 h 开发事件 64 个 0、留出 63 个 0；实验 h 全 1；AUROC 无定义，低 Brier 没有区分价值。','complete_evidence.json'),
      ('3.4','独立患者概率校准 A','数据缺失','没有可用的独立、身份核实的 A 组。未拟合或发布真实概率校准器。','certification_audit.json'),
      ('3.5','预设风险认证 B','数学公式可核验，实证保证未成立','固定 7 阈值、alpha=0.05、预算=0.10；即使零错误也至少需要 47 个被接受的独立 B 组；当前 0 个核实患者组。','certification_audit.json；tests.txt'),
      ('3.6','H&E-only 自动拒绝','实际入口实测通过','监测了文件图像读取；只读输入 H&E。64 窗可信放行 0；拒绝不当作阴性。门控权重摘要每文件仅计算一次。','he_only_access_audit.json；he_only_inference/prediction.json')]
    bugs=read(base/'issues.json')['fixed']
    bug_proofs={
      'B1':'旧质控独立重算与旧 manifest 完全一致；新有效窗数量由完整清单复算。',
      'B2':'旧版改名最大 RGB 差 0.03205；当前改名样本及方差逐元素相等，测试通过。',
      'B3':'旧版无关 argv 导致 exit=2；当前同样 argv 不污染工厂，测试通过。',
      'B4':'256×256 prepare→选项→数据读取完整测试通过。',
      'B5':'记录模式保留图像对象为 0，回调仍逐窗执行，测试通过。',
      'B6':'两实例缓存隔离、容量上限及重复访问命中，测试通过。',
      'B7':'越界/NaN 分数、旧协议、权重绑定和越风险上界证书均拒绝，测试通过。',
      'B8':'实际 H&E 推理记录：门控 G/h 摘要各调用 1 次，不随 64 窗重复。',
      'B9':'大图 PNG 无损往返测试通过；所有随本报告图像解码通过。'}
    bug_rows=[(b[0],b[1],bug_proofs[b[0]]) for b in bugs]
    bug_rows.append(('B10','固定 top-16 漏检形态冲突','旧 88 对；全半径 106 对；独立穷举恰好 106 对。密集合成簇 400 对完整枚举回归测试通过。'))
    tests=(out/'tests.txt').read_text();assert 'Ran 32 tests' in tests and '\nOK' in tests
    for png in out.rglob('*.png'):
        with Image.open(png) as im:im.load()
    (out/'step_evidence_complete.json').write_text(json.dumps([dict(step=r[0],action=r[1],status=r[2],evidence=r[3],source=r[4]) for r in steps],ensure_ascii=False,indent=2)+'\n')
    proof_index={str(p.relative_to(root)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'EVIDENCE_INDEX.json','complete_evidence_report.html','complete_evidence_report.md'}}
    for p in [root/'expression/challenge.py',root/'tools/audit_complete_evidence.py',root/'tools/report_complete_evidence.py',root/'tests/test_proof_audit.py']:
        proof_index[str(p.relative_to(root))]=sha256(p)
    (out/'EVIDENCE_INDEX.json').write_text(json.dumps(proof_index,ensure_ascii=False,indent=2)+'\n')
    def figure(name,caption):
        return '<figure><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'
    style='body{font:16px/1.7 system-ui,sans-serif;color:#20303c;background:#f4f6f7;margin:0}main{max-width:1220px;margin:auto;padding:28px 22px}h1{font-size:32px;line-height:1.3}h2{font-size:23px;margin-top:36px}p{max-width:1050px}.box{padding:16px 20px;background:white;border-left:5px solid #b85635;border-radius:6px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.card{background:white;padding:17px;border-top:4px solid #b85635}.card b{display:block}figure{margin:26px 0;padding:12px;background:white}img{display:block;width:100%;height:auto}figcaption{font-size:14px;color:#465c68;padding:8px}.table{overflow:auto;background:white}table{border-collapse:collapse;width:100%;min-width:760px}th,td{text-align:left;vertical-align:top;padding:11px 12px;border-bottom:1px solid #dbe2e5}th{background:#e7eef0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:white;padding:16px}details{margin:12px 0;background:#fff;padding:12px}code{background:#e7eef0;padding:2px 4px}small{color:#526977}@media(max-width:700px){.cards{grid-template-columns:1fr}h1{font-size:26px}main{padding:18px 12px}}'
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CUT 全流程逐条证据审计</title><style>'+style+'</style></head><body><main>'
    page+='<h1>逐条证明到哪里了？<br>三项创新、18 步实现、10 项软件修复的证据账本</h1>'
    page+='<p class="box"><b>结论：不能据现有数据证明三项创新全部有效。</b> 本次对真实附件、冻结权重和原始记录独立复算；软件流程与拒绝入口可以验证，表达边界、真实后验和患者级校准均未成立。下面保留反证和证据缺口，不把“能运行”写成“有效”。</p>'
    page+='<p>数据：新增 6 对＋旧 1 对，7 张 H&E 和 7 张 IHC；448 个原始同坐标窗、397 个规则有效窗；训练／开发／留出为 270／64／63 窗。图片角色沿用之前的固定划分；两次 G 训练各 160 步、ngf=8、6 残差块。本次没有重新训练 G、没有使用留出结果选择新 checkpoint，也没有虚构患者标识。</p>'
    page+='<div class="cards"><div class="card"><b>创新 ① 可预测边界：未证明</b>106 对含训练图的代理候选；开发／留出候选为 0；没有核实的蛋白相反真值。</div><div class="card"><b>创新 ② 表达后验：未证明</b>分布预测弱于训练频率先验；定位和条件控制失败；细胞级概率尚未实现。</div><div class="card"><b>创新 ③ 可信校准：未证明</b>事件只有一类；独立 A／B 数据缺失；可信展示为 0／64。</div></div>'
    page+=figure('evidence_summary.png','独立复算与负对照。左：新发现的检索漏检已修复；中：预测头没有通过简单对照；右：条件响应远不足以跟随指定表达量。图中两个常数对照均仅使用训练标签计算。')
    page+='<h2>18 个实现步骤逐条核对</h2>'+table(['步骤','实现','结论等级','实际证据','原始文件'],steps)
    page+='<h2>真实图片与原位参考：自动选窗</h2>'+figure('actual_window_evidence.png','自动选择留出图中参考 DAB 比例最高的窗。选窗规则用于诊断展示，全部 63 窗才用于量化。像素头是 DAB 代理，不是细胞蛋白阳性概率。')
    page+='<h2>挑战候选：可复现，也有明确边界</h2>'+figure('automatic_candidate_evidence.png','选择距离最小的合格候选，无人工 ROI。H&E 在不同窗口之间做形态比较；每个窗口的 IHC 始终读取它自己的原坐标，未做区域外对应搜索。')
    metric_rows=[['固定零面积率',f"{m['zero_baseline_mae']:.6f}",'—','—'],['默认兼容 G',f"{m['area_mae']:.6f}",f"{m['mean_dab_iou']:.6f}",f"{m['rgb_l1']:.6f}"],['失败实验组合（默认关闭）',f"{ex['area_mae']:.6f}",f"{ex['mean_dab_iou']:.6f}",f"{ex['rgb_l1']:.6f}"]]
    page+='<h2>独立重新计算，而非抄汇总表</h2><p>重新加载保存的 G／h，对 63 个留出窗推理，再用独立 NumPy float64 实现测量原始浮点图像。未从显示 PNG 反算指标。与旧记录的最大数值差为 '+f"{max(m['max_independent_metric_difference_from_saved'],ex['max_independent_metric_difference_from_saved']):.3g}"+'，小于 10⁻⁶。兼容版本与原 v1 的 G 参数逐元素差为 0。</p>'+table(['方法','DAB 面积率 MAE','DAB 像素 IoU','RGB L1（[-1,1]）'],metric_rows)
    page+='<p>全部留出窗都在第 0 分箱，100% 分箱准确率没有区分价值。训练频率先验仅由训练窗计数、加一平滑得到。像素常数基线也只使用训练窗的组织内阳性率。另保存了使用留出患病率的诊断下限，并未将其当作部署比较器。本次未补做大规模训练或因果消融，所以也不能证明是哪一个实验组件导致退步。</p>'
    page+='<h2>错误预测、拒绝与数学上界</h2>'+figure('risk_and_certification_evidence.png','左图只有同来源图像内诊断意义，不是患者级风险曲线；IoU<0.1 是额外空间诊断，未被事后替换为 h 的训练事件。右图是软件公式检验，不是这些附件得到的认证。')
    page+='<p>零错误时，上界 <code>U=1−(α/K)^(1/n)</code>。这里 K=7、α=0.05；预算 0.10 下，46 个已接受的独立 B 组仍不足，47 个才刚满足零错误条件。非零错误需要更多证据。一般使用固定候选阈值的 Bonferroni 修正单侧二项上界。条件于已冻结的 A 校准器，每个候选的越界概率至多 α/K，联合界至多 α；这依赖真正独立且同分布的来源组，不认证所有相关图像窗或所有像素。</p>'
    page+='<p>当前数据不满足这些条件；实际来源重叠被程序拒绝。A 校准的拟合本身也不提供有限样本概率校准保证。HE-only 推理的文件读取已监测：输入只读取 H&E；输出 PNG 的解码核验另行记录；G/h 门控摘要各计算 1 次。64 窗均未放行，拒绝不代表表达阴性。</p>'
    page+='<h2>10 项软件问题：每项对应证据</h2>'+table(['编号','问题','本次可核对依据'],bug_rows)
    page+='<h2>还差什么才有机会证明研究有效</h2>'+table(['创新','不可省略的数据或实验','预先确定的通过方式'],[
        ['① 边界','确认标记物和患者/切片身份；独立测试来源；对形态近似冲突候选核实真实表达','冻结距离/表达冲突规则；按来源报告挑战性能、正常集性能及固定覆盖率错误率'],
        ['② 后验','足量且表达范围完整的独立实染；可靠细胞/膜定位真值；完整 CUT 与单模块消融、多种子训练','独立分布评分胜过训练先验；空间/细胞指标达预设标准；概率与样本覆盖得到外部验证'],
        ['③ 校准','冻结 G/h 与临床有意义的错误定义；同时含正确/错误事件的独立 risk_train、A、B、test','AUC/适当评分与校准误差；独立 B 上界低于预算；锁定 test 报告实际风险和覆盖率']])
    page+='<p>这些要求不能通过把同一张图切得更碎、生成更多样本、伪造患者编号或事后改变阈值替代。像素 IoU 以给定配对为参考，可能受残余错位或切片关系影响；在配准精度未核实前，不能当作细胞级准确性。未确认标记物的 IHC 也不能支持“标记物特异”或 HER2 临床评分结论。技术有效性实验通过后，首次创新与否仍需独立文献和现有方法比较，本报告没有证明首次创新。</p>'
    page+='<h2>一键复核与完整记录</h2><pre>python -m unittest tests.test_expression tests.test_multipair_debug tests.test_proof_audit -v\npython -m tools.audit_complete_evidence\npython -m tools.report_complete_evidence</pre><p>第一行执行 32 项测试；第二行读取真实图片和冻结权重，独立重算，不重新训练 G；第三行生成本报告。原始训练复现入口仍为 <code>python tools/run_multipair_debug.py --steps 160</code>。完整证据目录为 <code>artifacts/complete_evidence</code>；图片、权重和原始训练日志保留在代码包内。EVIDENCE_INDEX.json 用于核对本次证据文件 SHA256。</p>'
    page+='<details><summary>展开本次 32 项测试原始输出</summary><pre>'+html.escape(tests)+'</pre></details><details><summary>展开独立复算与证书审计 JSON</summary><pre>'+html.escape(json.dumps(d,ensure_ascii=False,indent=2))+'</pre></details></main></body></html>'
    (out/'complete_evidence_report.html').write_text(page)
    md='# CUT 全流程证据审计\n\n三项创新均未得到完整有效性证明。软件 32 项回归测试通过；发现并修复第 10 项问题：top-16 漏掉 18 个符合固定规则的候选。\n\n'
    md+='| 步骤 | 实现 | 结论 | 证据 | 文件 |\n|---|---|---|---|---|\n'+''.join('| '+' | '.join(r)+' |\n' for r in steps)
    md+='\n本次 G 未重新训练；重新加载实际权重独立推理、独立 NumPy 计量。全部同坐标；无人工 ROI；来源身份未核实，不能声称患者级泛化。详见自包含 HTML 和 complete_evidence.json。\n'
    (out/'complete_evidence_report.md').write_text(md)
    print(json.dumps(dict(report=str(out/'complete_evidence_report.html'),tests=32,steps=len(steps),software_issues=len(bug_rows)),ensure_ascii=False))


if __name__=='__main__':main()

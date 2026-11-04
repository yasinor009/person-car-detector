"""Measured release criteria. Model confidence must never substitute for these metrics."""
import math


def metric_gate(metrics,target=.9,min_objects=50):
    result={}
    for name in ('person','car'):
        row=metrics.get('per_class',{}).get(name)
        if row is None:
            result[name]={'passed':False,'reason':'missing_class_metrics'};continue
        objects=row['true_positives']+row['false_negatives']
        predictions=row['true_positives']+row['false_positives']
        precision=row['true_positives']/predictions if predictions else 0.
        recall=row['true_positives']/objects if objects else 0.
        passed=(objects>=min_objects and predictions>0 and precision>=target and recall>=target)
        result[name]={'passed':passed,'precision':precision,'recall':recall,
                      'objects':objects,'predictions':predictions,'precision_wilson95':row.get('precision_wilson95')}
    return dict(passed=all(x['passed'] for x in result.values()),classes=result)


def acceptance_report(metrics,split='development',long_range_metrics=None,verified_range_m=None,
                      inference_fps=None,min_fps=10.,target=.9,min_objects=50):
    if split not in ('development','validation','test'):raise ValueError('Invalid split')
    if not 0<target<1 or min_objects<1:raise ValueError('Invalid acceptance criteria')
    general=metric_gate(metrics,target,min_objects)
    if long_range_metrics is not None and verified_range_m is not None and verified_range_m>=150:
        far=metric_gate(long_range_metrics,target,min_objects)
        far['evidence']='user-supplied measured-distance image manifest'
    else:far={'passed':False,'reason':'no_independent_measured_150m_test_images'}
    speed=bool(inference_fps is not None and math.isfinite(inference_fps) and inference_fps>=min_fps)
    inference_fps=float(inference_fps) if inference_fps is not None else None
    blockers=[]
    if split!='test':blockers.append('Dataset is not an independent test split')
    if not general['passed']:blockers.append('Both precision and recall >=90% are not established for each class with sufficient objects')
    if not far['passed']:blockers.append('Both class targets at measured >=150m are not established')
    if not speed:blockers.append(f'Measured inference throughput >= {min_fps:g} FPS is not established')
    return dict(passed=not blockers,target_precision=target,target_recall=target,minimum_objects_per_class=min_objects,
                split=split,general=general,long_range=far,speed_passed=speed,inference_fps=inference_fps,
                minimum_inference_fps=min_fps,blockers=blockers,
                scope='Point metrics on this dataset only; inference FPS excludes capture/rendering; correlated data and domain shift may reduce generalization')

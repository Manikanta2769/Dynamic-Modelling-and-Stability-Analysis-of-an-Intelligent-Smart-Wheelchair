"""Plots for the 3-D wheelchair model."""
import os
import matplotlib.pyplot as plt
import numpy as np
from parameters import *
WHEEL_LABELS = ['Front right (passive)', 'Front left (passive)', 'Rear right (driven)', 'Rear left (driven)']
os.makedirs(GRAPH_DIR, exist_ok=True)


def _tag_terrain(r):
    return f" (grade {r['grade_deg'][0]:g}°, cross-slope {r['cross_slope_deg'][0]:g}°)"


def _tag(r):
    return (f" ({r.get('controller_name', 'controller')}, {r.get('scenario', 'scenario')})"
            + _tag_terrain(r))


def _filename(r, stem):
    """Use stable filenames so each run refreshes the same graph.

    Scenario and terrain remain in the filename to keep genuinely different
    test conditions separate. Controller name is intentionally omitted:
    rerunning a controller comparison updates the same graph instead of
    accumulating one file per controller.
    """
    scenario = str(r.get('scenario', 'scenario')).strip().replace(' ', '_')
    grade = f"{float(r['grade_deg'][0]):g}".replace('-', 'm').replace('.', 'p')
    cross = f"{float(r['cross_slope_deg'][0]):g}".replace('-', 'm').replace('.', 'p')
    return f"{stem}_{scenario}_g{grade}_x{cross}.png"


def save_plot(x,y,xlabel,ylabel,title,filename,results=None):
    plt.figure(figsize=figure_size); plt.plot(x,y,linewidth=2)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.grid(True)
    if results is not None:
        filename = _filename(results, os.path.splitext(filename)[0])
    plt.tight_layout(); plt.savefig(GRAPH_DIR+filename,dpi=dpi); plt.close()


def plot_position(r):
    fig,ax=plt.subplots(figsize=figure_size)
    ax.plot(r['time'],r['position_xyz'][:,0],label='x'); ax.plot(r['time'],r['position_xyz'][:,1],label='y'); ax.plot(r['time'],r['position_xyz'][:,2],label='z')
    ax.set(xlabel='Time (s)',ylabel='World position (m)',title='3-D Position Components'+_tag(r)); ax.legend(); ax.grid(True); fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'position_3d_components')),dpi=dpi); plt.close(fig)


def plot_velocity(r):
    fig,ax=plt.subplots(figsize=figure_size)
    ax.plot(r['time'],r['velocity_x'],label='vx'); ax.plot(r['time'],r['velocity_y'],label='vy'); ax.plot(r['time'],r['velocity_z'],label='vz'); ax.plot(r['time'],r['velocity'],label='speed magnitude',linewidth=2); ax.plot(r['time'],r['velocity_forward'],label='forward along terrain'); ax.plot(r['time'],r['reference_speed'],'k--',label='reference speed')
    ax.set(xlabel='Time (s)',ylabel='Velocity (m/s)',title='3-D Velocity'+_tag(r)); ax.legend(); ax.grid(True); fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'velocity_3d')),dpi=dpi); plt.close(fig)


def plot_acceleration(r):
    fig,ax=plt.subplots(figsize=figure_size)
    ax.plot(r['time'],r['acceleration_x'],label='ax'); ax.plot(r['time'],r['acceleration_y'],label='ay'); ax.plot(r['time'],r['acceleration_z'],label='az'); ax.plot(r['time'],r['acceleration'],label='|a|',linewidth=2)
    ax.set(xlabel='Time (s)',ylabel='Acceleration (m/s²)',title='3-D Acceleration'+_tag(r)); ax.legend(); ax.grid(True); fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'acceleration_3d')),dpi=dpi); plt.close(fig)


def plot_omega(r):
    fig,ax=plt.subplots(figsize=figure_size)
    ax.plot(r['time'],r['wheel_speed'],label=WHEEL_LABELS); ax.legend(); ax.set(xlabel='Time (s)',ylabel='Wheel speed (rad/s)',title='Four Wheel Angular Speeds'+_tag(r)); ax.grid(True); fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'wheel_angular_velocity')),dpi=dpi); plt.close(fig)


def plot_torque(r):
    fig,ax=plt.subplots(figsize=figure_size)
    ax.plot(r['time'],r['torque_wheels'],label=WHEEL_LABELS); ax.legend(); ax.set(xlabel='Time (s)',ylabel='Wheel torque (Nm)',title='Individual Wheel Torques'+_tag(r)); ax.grid(True); fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'wheel_torque_3d')),dpi=dpi); plt.close(fig)


def plot_energy(r):
    fig,ax=plt.subplots(figsize=figure_size)
    ax.plot(r['time'],r['kinetic_energy'],label='Kinetic'); ax.plot(r['time'],r['potential_energy'],label='Gravity potential'); ax.plot(r['time'],r['contact_elastic_energy'],label='Contact spring'); ax.plot(r['time'],r['attitude_elastic_energy'],label='Attitude spring'); ax.plot(r['time'],r['total_energy'],label='Stored mechanical energy',linewidth=2)
    ax.set(xlabel='Time (s)',ylabel='Energy (J)',title='3-D Mechanical Energy'+_tag(r)); ax.legend(); ax.grid(True); fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'energy_3d')),dpi=dpi); plt.close(fig)


def plot_power(r): save_plot(r['time'],r['power'],'Time (s)','Mechanical power (W)','Wheel Mechanical Power','power_3d.png',r)

def plot_stability(r):
    fig, ax = plt.subplots(figsize=figure_size)
    traction_ax = ax.twinx()
    ax.plot(r['time'], r['support_margin_min'], color='tab:blue', label='Support margin')
    traction_ax.plot(r['time'], 1-r['friction_utilization_3d'],
                     color='tab:orange', label='Residual traction capacity')
    ax.axhline(0, color='tab:blue', linestyle='--', alpha=0.7)
    traction_ax.axhline(0, color='tab:orange', linestyle=':', alpha=0.7)
    ax.set(xlabel='Time (s)', ylabel='Support margin (m)',
           title='Quasi-static Support and Traction Screens'+_tag(r))
    traction_ax.set_ylabel('Residual traction capacity (dimensionless)')
    lines = ax.get_lines()[:1] + traction_ax.get_lines()[:1]
    ax.legend(lines, [line.get_label() for line in lines], loc='best')
    ax.grid(True)
    fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'stability_3d')),dpi=dpi); plt.close(fig)


def plot_phase(r): save_plot(r['position'],r['velocity_forward'],'Forward distance (m)','Forward velocity (m/s)','Forward Motion Phase Portrait','phase_3d.png',r)


def plot_trajectory_3d(r):
    fig=plt.figure(figsize=figure_size); ax=fig.add_subplot(111,projection='3d')
    p=r['position_xyz']; ax.plot(p[:,0],p[:,1],p[:,2],linewidth=2); ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Wheelchair 3-D Trajectory'+_tag(r)); fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'trajectory_3d')),dpi=dpi); plt.close(fig)


def plot_attitude(r):
    fig,ax=plt.subplots(figsize=figure_size)
    ax.plot(r['time'],r['roll'],label='Roll',linewidth=2.5); ax.plot(r['time'],r['pitch'],label='Pitch'); ax.plot(r['time'],r['yaw'],label='Yaw',linestyle='--')
    ax.axhline(0.0,color='k',linestyle=':',linewidth=1,label='terrain-aligned attitude')
    ax.set(xlabel='Time (s)',ylabel='Angle relative to terrain (deg)',title='Body Attitude Relative to Terrain'+_tag(r)); ax.legend(loc='upper center',ncol=3,fontsize=8); ax.grid(True); fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'attitude_3d')),dpi=dpi); plt.close(fig)


def plot_normal_loads(r):
    fig,ax=plt.subplots(figsize=figure_size)
    ax.plot(r['time'],r['normal_loads'],label=WHEEL_LABELS); ax.legend(); ax.set(xlabel='Time (s)',ylabel='Normal load (N)',title='Four Wheel Normal Loads'+_tag(r)); ax.grid(True); fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'wheel_normal_loads')),dpi=dpi); plt.close(fig)


def plot_controller_comparison(comparison):
    fig,ax=plt.subplots(figsize=figure_size)
    for controller,data in comparison.items(): ax.plot(data['time'],data['position']-data['position'][0],label=controller.upper())
    first=next(iter(comparison.values())); ax.plot(first['time'],first['reference'],'k--',label='Reference'); ax.legend(); ax.grid(True); ax.set(xlabel='Time (s)',ylabel='Distance along terrain (m)',title='3-D Controller Comparison'); fig.tight_layout(); fig.savefig(GRAPH_DIR+'controller_comparison.png',dpi=dpi); plt.close(fig)


def plot_scenario_comparisons(experiments, grade=None, cross=None):
    for scenario,comparison in experiments.items():
        for _data in comparison.values():
            print(f"  saving graphs: {scenario} / {_data.get('controller_name')} ...", flush=True)
            plot_controller_graphs(_data, subfolder=scenario)
    print('All controller graphs saved under:', CONTROLLER_GRAPH_DIR)
    return  # legacy overlay figures below are intentionally no longer produced
    for scenario,comparison in experiments.items():
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
        LS={'pid':'-','nonlinear_pid':'--','fuzzy':':','smc':'-.'}
        for controller,data in comparison.items():
            ls=LS.get(controller,'-'); lab=controller.upper()
            axes[0,0].plot(data['time'],data['position']-data['position'][0],label=lab,linestyle=ls)
            axes[0,1].plot(data['time'],data['velocity_forward'],label=lab,linestyle=ls)
            axes[1,0].plot(data['time'],data['roll'],label=lab,linestyle=ls)
            axes[1,1].plot(data['time'],data['pitch'],label=lab,linestyle=ls)
        ref=next(iter(comparison.values()))
        axes[0,0].plot(ref['time'],ref['reference'],'k:',label='Reference'); axes[0,1].plot(ref['time'],ref['reference_speed'],'k:',label='Reference')
        axes[0,0].set_ylabel('Distance along terrain (m)'); axes[0,1].set_ylabel('Speed (m/s)'); axes[1,0].set_ylabel('Roll relative to terrain (deg)'); axes[1,1].set_ylabel('Pitch relative to terrain (deg)')
        axes[1,0].set_xlabel('Time (s)'); axes[1,1].set_xlabel('Time (s)'); axes[0,0].legend(); axes[0,1].legend(fontsize=8);
        for ax in axes.flat: ax.grid(True)
        axes[1,0].axhline(0.0, color='k', linestyle=':', linewidth=0.8)
        axes[1,1].axhline(0.0, color='k', linestyle=':', linewidth=0.8)
        fig.suptitle(f'3-D controller responses: {scenario}' + (f' (grade {grade:g} deg, cross-slope {cross:g} deg)' if grade is not None else '')); fig.tight_layout(); fig.savefig(GRAPH_DIR+(f'comparison_{scenario}_grade{grade:g}_cross{cross:g}.png' if grade is not None else f'comparison_{scenario}.png'),dpi=dpi); plt.close(fig)


def plot_ml_gain_scales(r):
    if 'ml_gain_scales' not in r:
        return
    fig, ax = plt.subplots(figsize=figure_size)
    for name, values in zip(r['ml_gain_names'], r['ml_gain_scales'].T):
        ax.plot(r['ml_gain_time'], values, label=name)
    ax.set(xlabel='Time (s)', ylabel='Predicted gain multiplier',
           title='Online ML Gain Scheduling')
    ax.grid(True); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(GRAPH_DIR,_filename(r,'ml_gain_scales')), dpi=dpi); plt.close(fig)


CONTROLLER_GRAPH_DIR = os.path.join('output', 'controller_graphs')
_PRETTY = {'pid': 'Linear PID', 'nonlinear_pid': 'Nonlinear PID', 'fuzzy': 'Fuzzy', 'smc': 'Sliding Mode'}
_AX = ['x', 'y', 'z']


def plot_controller_graphs(r, subfolder=None):
    """Per-controller summary graph (00_summary.png) -- the only graph produced.

    Saved to output/controller_graphs/<controller>/00_summary.png
    (inside output/controller_graphs/<subfolder>/<controller>/ when a subfolder is given).
    Panels: speed vs reference, position x/y/z, velocity x/y/z, acceleration x/y/z,
    total contact force x/y/z, drive force + disturbance, roll/pitch.
    """
    t = r['time']; name = r.get('controller_name', 'controller'); scen = str(r.get('scenario', 'scenario'))
    gain = 'ML-adaptive gains' if 'ml_gain_scales' in r else 'fixed gains'
    cf_total = r['contact_forces'].sum(axis=1)
    drive_force = r['torque_wheels'].sum(axis=1) / wheel_radius
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(4, 2)
    ax_speed = fig.add_subplot(gs[0, 0]); ax_pos = fig.add_subplot(gs[0, 1], sharex=ax_speed)
    ax_vel = fig.add_subplot(gs[1, 0], sharex=ax_speed); ax_acc = fig.add_subplot(gs[1, 1], sharex=ax_speed)
    ax_cf = fig.add_subplot(gs[2, 0], sharex=ax_speed); ax_drv = fig.add_subplot(gs[2, 1], sharex=ax_speed)
    ax_att = fig.add_subplot(gs[3, :], sharex=ax_speed)
    ax_speed.plot(t, r['velocity_forward'], label='forward speed'); ax_speed.plot(t, r['reference_speed'], 'k--', label='reference')
    ax_speed.set_title('Speed (m/s)')
    for i in range(3):
        ax_pos.plot(t, r['position_xyz'][:, i], label=f'p{_AX[i]}')
        ax_vel.plot(t, r['velocity_xyz'][:, i], label=f'v{_AX[i]}')
        ax_acc.plot(t, r['acceleration_xyz'][:, i], label=f'a{_AX[i]}')
        ax_cf.plot(t, cf_total[:, i], label=f'\u03a3F{_AX[i]}')
    ax_pos.set_title('Position components px, py, pz (m)')
    ax_vel.set_title('Velocity components (m/s)'); ax_acc.set_title('Acceleration components (m/s\u00b2)')
    ax_cf.set_title('Total contact force components (N)')
    ax_drv.plot(t, drive_force, label='drive force'); ax_drv.plot(t, r['disturbance_force'], label='|disturbance|')
    ax_drv.set_title('Drive / disturbance force (N)')
    ax_att.plot(t, r['roll'], label='roll'); ax_att.plot(t, r['pitch'], label='pitch'); ax_att.set_title('Roll / pitch (deg)')
    for a_ in fig.axes: a_.grid(True); a_.legend(fontsize=8)
    ax_drv.set_xlabel('Time (s)'); ax_cf.set_xlabel('Time (s)'); ax_att.set_xlabel('Time (s)')
    fig.suptitle(f"Summary\n{_PRETTY.get(name, name)} | {scen}" + _tag_terrain(r) + f" | {gain}", fontsize=12)
    folder = os.path.join(CONTROLLER_GRAPH_DIR, *( [subfolder] if subfolder else [] ), name); os.makedirs(folder, exist_ok=True)
    fig.tight_layout()
    target = os.path.join(folder, '00_summary.png')
    try:
        fig.savefig(target, dpi=dpi)
    except OSError as exc:
        # Windows can refuse to overwrite a file that is open in an image viewer,
        # preview pane, OneDrive, or antivirus scan. Keep the graph under a new name.
        import time as _time
        alt = os.path.join(folder, f"00_summary_{_time.strftime('%Y%m%d_%H%M%S')}.png")
        print(f"  [WARNING] could not overwrite {target} ({exc}); saved {alt} instead. "
              "Close any program that has the old image open.", flush=True)
        fig.savefig(alt, dpi=dpi)
    plt.close(fig)


def plot_pybullet_summary(d, controller_name, ml_adaptive=True, out_root=CONTROLLER_GRAPH_DIR):
    """Summary graph for the PyBullet terrain run (speed humps, 10 deg climb, crest, 10 deg decline).

    Everything is derived from the recorded POSITION so the panels agree with each other:
        position p(t)  ->  velocity v = dp/dt  ->  acceleration a = dv/dt  ->  force F = m (a - g)
        speed = |v|,  path slope = atan(vz / vx)  (both from the position record)
    No target / reference speed is plotted: the speed is whatever the controller achieves.
    d: dict with mass (kg), time, position (N,3), drive_force, roll, pitch (PyBullet convention).
    Saved to <out_root>/<controller key>/00_summary.png
    """
    key = {'Linear PID': 'pid', 'Nonlinear PID': 'nonlinear_pid', 'Sliding Mode': 'smc', 'Fuzzy': 'fuzzy'}.get(controller_name, controller_name)
    t = np.asarray(d['time'], float); pos = np.asarray(d['position'], float)
    dt_ = float(np.median(np.diff(t)))

    def smooth(a, seconds):                       # moving average, edges held (no end artefacts)
        w = max(1, int(round(seconds / dt_)))
        if w == 1: return a
        pad = (w // 2, w - 1 - w // 2); k = np.ones(w) / w
        a = np.asarray(a, float)
        cols = [a] if a.ndim == 1 else [a[:, i] for i in range(a.shape[1])]
        out = [np.convolve(np.pad(c, pad, mode='edge'), k, mode='valid') for c in cols]
        return out[0] if a.ndim == 1 else np.column_stack(out)

    p_s = smooth(pos, 0.10)
    vel = smooth(np.column_stack([np.gradient(p_s[:, i], t) for i in range(3)]), 0.05)
    acc = smooth(np.column_stack([np.gradient(vel[:, i], t) for i in range(3)]), 0.10)
    cf = float(d['mass']) * (acc + np.array([0.0, 0.0, 9.81]))          # Newton: sum of ground forces = m(a - g)
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > 0.15
    path_slope = np.where(moving, np.degrees(np.arctan2(vel[:, 2], np.where(moving, vel[:, 0], 1.0))), np.nan)
    z0 = float(np.mean(pos[:max(1, int(0.5 / dt_)), 2]))               # resting height of the chair at the start
    drive = smooth(np.asarray(d['drive_force'], float), 0.05)
    pitch_up = -np.asarray(d['pitch'], float)                            # PyBullet pitch is negative nose-up; plot nose-up as +

    fig = plt.figure(figsize=(14, 13)); gs = fig.add_gridspec(4, 2)
    ax_pos = fig.add_subplot(gs[0, 0]); ax_speed = fig.add_subplot(gs[0, 1], sharex=ax_pos)
    ax_vel = fig.add_subplot(gs[1, 0], sharex=ax_pos); ax_acc = fig.add_subplot(gs[1, 1], sharex=ax_pos)
    ax_cf = fig.add_subplot(gs[2, 0], sharex=ax_pos); ax_drv = fig.add_subplot(gs[2, 1], sharex=ax_pos)
    ax_att = fig.add_subplot(gs[3, :], sharex=ax_pos)
    ax_z = ax_pos.twinx()                       # pz gets its own scale so the 0.5 m climb is readable next to 12 m of travel
    ax_pos.plot(t, pos[:, 0], color='tab:blue', label='px (left scale)')
    ax_pos.plot(t, pos[:, 1], color='tab:orange', label='py (left scale)')
    ax_z.plot(t, pos[:, 2] - z0, color='tab:green', label='pz = height gained (right scale)')
    ax_pos.set_ylabel('px, py (m)'); ax_z.set_ylabel('pz (m)', color='tab:green')
    for i in range(3):
        ax_vel.plot(t, vel[:, i], label=f'v{_AX[i]}'); ax_acc.plot(t, acc[:, i], label=f'a{_AX[i]}')
        ax_cf.plot(t, cf[:, i], label=f'\u03a3F{_AX[i]}')
    ax_pos.set_title('Position components px, py, pz (m)')
    ax_speed.plot(t, speed, label='speed |v| (set by the controller)'); ax_speed.set_title('Speed (m/s)')
    ax_vel.set_title('Velocity components = d(position)/dt (m/s)')
    ax_acc.set_title('Acceleration components = d(velocity)/dt (m/s\u00b2)')
    ax_cf.set_title('Total ground contact force = m(a - g) (N)')
    ax_drv.plot(t, drive, label='drive force (motor torque / wheel radius)'); ax_drv.set_title('Drive force (N)')
    ax_att.plot(t, d['roll'], label='roll'); ax_att.plot(t, pitch_up, label='chair pitch (nose-up = +)')
    ax_att.plot(t, path_slope, 'k:', linewidth=2, label='slope of the path from position = atan(dz/dx)')
    ax_att.set_title('Roll / pitch (deg)')
    for a_ in fig.axes:
        a_.grid(True)
        if a_ is not ax_z: a_.legend(fontsize=8)
    h1, l1 = ax_pos.get_legend_handles_labels(); h2, l2 = ax_z.get_legend_handles_labels()
    ax_pos.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    for a_ in (ax_drv, ax_cf, ax_att): a_.set_xlabel('Time (s)')
    gain = 'ML-adaptive gains' if ml_adaptive else 'fixed gains'
    fig.suptitle(f"Summary (PyBullet terrain: 2 speed humps, 10\u00b0 climb, crest, 10\u00b0 decline)\n{controller_name} | {gain}", fontsize=12)
    folder = os.path.join(out_root, key); os.makedirs(folder, exist_ok=True)
    fig.tight_layout(); target = os.path.join(folder, '00_summary.png')
    try:
        fig.savefig(target, dpi=dpi)
    except OSError as exc:
        import time as _time
        alt = os.path.join(folder, f"00_summary_{_time.strftime('%Y%m%d_%H%M%S')}.png")
        print(f"  [WARNING] could not overwrite {target} ({exc}); saved {alt} instead.", flush=True)
        fig.savefig(alt, dpi=dpi)
    plt.close(fig)
    return target


def generate_all_plots(results):
    # Only the per-controller component / acceleration / force graphs are produced.
    plot_controller_graphs(results)
    print('\nGraphs saved under:', CONTROLLER_GRAPH_DIR)
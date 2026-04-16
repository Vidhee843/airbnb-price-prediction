import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('final_data.csv')

# filter extreme price outliers
df = df[df['price'] < 2000].copy()


'''Chart 1 - Price Distribution'''

plt.figure(figsize=(9,5))
plt.hist(df['price'], bins=60, color='#2E75B6', edgecolor='white', alpha=0.85)
plt.axvline(df['price'].mean(), color='red', linestyle='--', linewidth=1.5,
            label=f"Mean: ${df['price'].mean():.0f}")
plt.axvline(df['price'].median(), color='orange', linestyle='--', linewidth=1.5,
            label=f"Median: ${df['price'].median():.0f}")
plt.title('Distribution of Nightly Prices', fontsize=14, fontweight='bold')
plt.xlabel('Price (USD/night)', fontsize=11)
plt.ylabel('Number of Listings', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('chart1_price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart1 done')


'''Chart 2 - Average Price by Room Type'''

room_map = {
    'room_type_Entire home/apt': 'Entire home/apt',
    'room_type_Private room':    'Private room',
    'room_type_Hotel room':      'Hotel room',
    'room_type_Shared room':     'Shared room'
}

room_avg = {}
for col, label in room_map.items():
    avg = df[df[col] == True]['price'].mean()
    room_avg[label] = round(avg, 2)

room_df = pd.Series(room_avg).sort_values(ascending=False)

plt.figure(figsize=(8,5))
bars = plt.bar(room_df.index, room_df.values,
               color=['#2E75B6','#70AD47','#ED7D31','#FFC000'],
               edgecolor='white', width=0.5)
for bar, val in zip(bars, room_df.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f'${val:.0f}', ha='center', fontsize=11, fontweight='bold')
plt.title('Average Nightly Price by Room Type', fontsize=14, fontweight='bold')
plt.ylabel('Average Price (USD/night)', fontsize=11)
plt.ylim(0, room_df.max() * 1.2)
plt.tight_layout()
plt.savefig('chart2_price_by_room_type.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart2 done')


'''Chart 3 - Median Price by Number of Bedrooms'''

bed_df = df[df['bedrooms'] <= 6].copy()
bed_avg = bed_df.groupby('bedrooms')['price'].median().round(2)
bed_labels = ['Studio' if b == 0 else str(int(b)) for b in bed_avg.index]

plt.figure(figsize=(9,5))
plt.plot(bed_labels, bed_avg.values, marker='o', linewidth=2.5,
         markersize=9, color='#2E75B6')
for x, y in zip(bed_labels, bed_avg.values):
    plt.annotate(f'${y:.0f}', (x, y), textcoords='offset points',
                 xytext=(0, 12), ha='center', fontsize=9, color='#2E75B6')
plt.title('Median Nightly Price by Number of Bedrooms', fontsize=14, fontweight='bold')
plt.xlabel('Number of Bedrooms', fontsize=11)
plt.ylabel('Median Price (USD/night)', fontsize=11)
plt.tight_layout()
plt.savefig('chart3_price_by_bedrooms.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart3 done')


'''Chart 4 - Price vs Distance from Downtown (Scatter)'''

plt.figure(figsize=(9,5))
plt.scatter(df['distance_km'], df['price'], alpha=0.2, color='#2E75B6',
            edgecolors='none', s=15)
z = np.polyfit(df['distance_km'], df['price'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['distance_km'].min(), df['distance_km'].max(), 100)
plt.plot(x_line, p(x_line), color='red', linewidth=2, label='Trend line')
plt.title('Price vs Distance from Downtown', fontsize=14, fontweight='bold')
plt.xlabel('Distance from Downtown (km)', fontsize=11)
plt.ylabel('Price (USD/night)', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('chart4_price_vs_distance.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart4 done')


'''Chart 5 - Top 10 Feature Importances (from Random Forest results)'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

y = df['price']
x = df.drop(columns='price')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

rfg = RandomForestRegressor(max_depth=14, min_samples_split=4, n_estimators=200, random_state=1)
rfg.fit(x_train, y_train)

feat_df = pd.DataFrame({
    'feature': x_train.columns,
    'importance': rfg.feature_importances_
}).sort_values('importance', ascending=False).head(10)
feat_df = feat_df.sort_values('importance')

plt.figure(figsize=(9,5))
bars = plt.barh(feat_df['feature'], feat_df['importance'],
                color='#2E75B6', edgecolor='white', alpha=0.85)
for bar, val in zip(bars, feat_df['importance']):
    plt.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=9)
plt.title('Top 10 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=11)
plt.tight_layout()
plt.savefig('chart5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart5 done')


'''Chart 6 - Average Price by Number of Amenities (Line)'''

df['amenities_bin'] = pd.cut(df['amenities_count'],
                              bins=[0,10,20,30,40,50,100],
                              labels=['1-10','11-20','21-30','31-40','41-50','50+'])
amenity_avg = df.groupby('amenities_bin', observed=True)['price'].mean().round(2)

plt.figure(figsize=(9,5))
plt.plot(amenity_avg.index.astype(str), amenity_avg.values,
         marker='o', linewidth=2.5, markersize=9, color='#70AD47')
plt.fill_between(range(len(amenity_avg)), amenity_avg.values,
                 alpha=0.15, color='#70AD47')
for x, y in enumerate(amenity_avg.values):
    plt.annotate(f'${y:.0f}', (x, y), textcoords='offset points',
                 xytext=(0, 12), ha='center', fontsize=9, fontweight='bold', color='#70AD47')
plt.xticks(range(len(amenity_avg)), amenity_avg.index.astype(str), fontsize=10)
plt.title('Average Price by Number of Amenities', fontsize=14, fontweight='bold')
plt.xlabel('Amenities Count (grouped)', fontsize=11)
plt.ylabel('Average Price (USD/night)', fontsize=11)
plt.ylim(0, amenity_avg.max() * 1.25)
plt.tight_layout()
plt.savefig('chart6_price_by_amenities.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart6 done')


'''Chart 7 - Review Score vs Price (Scatter)'''

plt.figure(figsize=(9,5))
plt.scatter(df['review_scores_rating'], df['price'], alpha=0.2,
            color='#ED7D31', edgecolors='none', s=15)
z = np.polyfit(df['review_scores_rating'], df['price'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['review_scores_rating'].min(), df['review_scores_rating'].max(), 100)
plt.plot(x_line, p(x_line), color='red', linewidth=2, label='Trend line')
plt.title('Review Score vs Nightly Price', fontsize=14, fontweight='bold')
plt.xlabel('Overall Review Score', fontsize=11)
plt.ylabel('Price (USD/night)', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('chart7_review_vs_price.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart7 done')


'''Chart 8 - Room Type Distribution (Bar Chart)'''

room_counts = {
    'Entire home/apt': int(df['room_type_Entire home/apt'].sum()),
    'Private room':    int(df['room_type_Private room'].sum()),
    'Hotel room':      int(df['room_type_Hotel room'].sum()),
    'Shared room':     int(df['room_type_Shared room'].sum())
}

total = sum(room_counts.values())
colors = ['#2E75B6','#70AD47','#ED7D31','#FFC000']

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(room_counts.keys(), room_counts.values(),
              color=colors, edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, room_counts.values()):
    pct = val / total * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_title('Room Type Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Listings', fontsize=11)
ax.set_ylim(0, max(room_counts.values()) * 1.25)
ax.set_xticklabels(room_counts.keys(), fontsize=10)
plt.tight_layout()
plt.savefig('chart8_room_type_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart8 done')


'''Chart 9 - Superhost vs Non-Superhost Average Price (Lollipop)'''

superhost_avg    = df[df['host_is_superhost_True']  == True]['price'].mean()
nonsuperhost_avg = df[df['host_is_superhost_False'] == True]['price'].mean()

categories = ['Not Superhost', 'Superhost']
values     = [nonsuperhost_avg, superhost_avg]
colors     = ['#ED7D31', '#2E75B6']

fig, ax = plt.subplots(figsize=(8, 5))
for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
    ax.plot([0, val], [i, i], color=color, linewidth=2.5, zorder=1)
    ax.scatter(val, i, color=color, s=300, zorder=2, edgecolors='white', linewidth=1.5)
    ax.text(val + 3, i, f'${val:.0f}/night', va='center',
            fontsize=12, fontweight='bold', color=color)

ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories, fontsize=12)
ax.set_xlabel('Average Price (USD/night)', fontsize=11)
ax.set_title('Average Price: Superhost vs Non-Superhost', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(values) * 1.3)
ax.axvline(x=0, color='gray', linewidth=0.8)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('chart9_superhost_price.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart9 done')


'''Chart 10 - Average Price by Bathroom Type (Donut Chart)'''

private_avg = df[df['bathroom type_Private'] == True]['price'].mean()
shared_avg  = df[df['bathroom type_shared']  == True]['price'].mean()

labels = ['Private Bathroom', 'Shared Bathroom']
values = [private_avg, shared_avg]
colors = ['#70AD47', '#FFC000']
total  = sum(values)

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts = ax.pie(values, labels=None, colors=colors,
                       startangle=90,
                       wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))

# center text
ax.text(0, 0, 'Avg Price\nComparison', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#2C3E50')

# clean outside labels — only 2 slices so no overlap
for wedge, label, val in zip(wedges, labels, values):
    angle = (wedge.theta1 + wedge.theta2) / 2
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))
    align = 'left' if x > 0 else 'right'
    ax.annotate(f'{label}\n${val:.0f}/night',
                xy=(x * 0.78, y * 0.78),
                xytext=(x * 1.2, y * 1.1),
                fontsize=11, fontweight='bold', ha=align, va='center',
                color=colors[labels.index(label)],
                arrowprops=dict(arrowstyle='-', color='gray', lw=1.0))

ax.set_title('Average Price by Bathroom Type', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('chart10_price_by_bathroom_type.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart10 done')


'''Chart 11 - Price Distribution by Room Type (Box Plot)'''

room_data = {
    'Entire\nhome/apt': df[df['room_type_Entire home/apt'] == True]['price'].values,
    'Private\nroom':    df[df['room_type_Private room']    == True]['price'].values,
    'Hotel\nroom':      df[df['room_type_Hotel room']      == True]['price'].values,
    'Shared\nroom':     df[df['room_type_Shared room']     == True]['price'].values,
}

colors = ['#2E75B6','#70AD47','#ED7D31','#FFC000']

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(room_data.values(), patch_artist=True,
                notch=False, widths=0.5,
                medianprops=dict(color='white', linewidth=2.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.3, linestyle='none'),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                boxprops=dict(linewidth=1.5))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax.set_xticklabels(room_data.keys(), fontsize=10)
ax.set_title('Price Distribution by Room Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Price (USD/night)', fontsize=11)
ax.set_ylim(0, 700)  # cap just above whiskers, not extreme outliers
plt.tight_layout()
plt.savefig('chart11_price_boxplot_by_roomtype.png', dpi=150, bbox_inches='tight')
plt.close()
print('chart11 done')


print('\nAll charts saved!')

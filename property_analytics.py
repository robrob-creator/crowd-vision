import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from firebase_admin import firestore

class PropertyAnalytics:
    """Property Management Analytics Dashboard"""

    def __init__(self, app_instance):
        self.app = app_instance

    def display(self):
        """Main analytics dashboard"""
        st.header("🏢 Property Management Analytics")

        if not st.session_state.sources:
            st.info("No property data available. Add sources in the 'Add Source' tab to begin collecting analytics.")
            return

        # Get all historical data
        analytics_data = self._collect_analytics_data()

        if not analytics_data:
            st.warning("No analytics data available yet. Start detection processes to begin collecting data.")
            return

        # Create tabs for different analytics views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview Dashboard",
            "🏘️ Property Performance",
            "🗑️ Waste Management",
            "👥 Occupancy Trends",
            "⏰ Peak Times & Patterns",
            "💡 Management Insights"
        ])

        with tab1:
            self._overview_dashboard(analytics_data)

        with tab2:
            self._property_performance(analytics_data)

        with tab3:
            self._waste_management(analytics_data)

        with tab4:
            self._occupancy_trends(analytics_data)

        with tab5:
            self._peak_times_analysis(analytics_data)

        with tab6:
            self._management_insights(analytics_data)

    def _collect_analytics_data(self):
        """Collect and aggregate analytics data from Realtime Database logs"""
        analytics_data = []

        try:
            # Get Realtime Database client from the app
            rt_db = self.app.rt_db

            for source_id, source in st.session_state.sources.items():
                location_name = source.get('location', 'Unknown')

                # Query Realtime Database for logs
                logs_ref = rt_db.reference('logs')
                logs_data = logs_ref.get()

                if not logs_data:
                    continue

                # Filter logs for this location
                location_logs = {}
                for log_key, log_data in logs_data.items():
                    if log_data.get('location') == location_name:
                        location_logs[log_key] = log_data

                if not location_logs:
                    continue

                # Initialize aggregation variables
                total_current_people = 0
                total_current_garbage = 0
                total_min_people = 0
                total_max_people = 0
                total_min_garbage = 0
                total_max_garbage = 0
                log_count = len(location_logs)
                latest_timestamp = None
                timestamps = []

                # Aggregate data from all logs for this location
                for log_key, data in location_logs.items():
                    # Aggregate current counts
                    total_current_people += data.get('currentNumberOfPeople', 0)
                    total_current_garbage += data.get('currentGarbage', 0)
                    
                    # Aggregate min/max counts
                    total_min_people += data.get('minPeopleCount', 0)
                    total_max_people += data.get('maxPeopleCount', 0)
                    total_min_garbage += data.get('minGarbageCount', 0)
                    total_max_garbage += data.get('maxGarbageCount', 0)

                    # Track timestamps
                    timestamp_str = data.get('lastUpdate', '')
                    if timestamp_str:
                        try:
                            # Parse ISO timestamp
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            timestamps.append(timestamp)
                            if latest_timestamp is None or timestamp > latest_timestamp:
                                latest_timestamp = timestamp
                        except:
                            pass

                # Calculate averages
                avg_current_people = total_current_people / max(log_count, 1)
                avg_current_garbage = total_current_garbage / max(log_count, 1)
                avg_min_people = total_min_people / max(log_count, 1)
                avg_max_people = total_max_people / max(log_count, 1)
                avg_min_garbage = total_min_garbage / max(log_count, 1)
                avg_max_garbage = total_max_garbage / max(log_count, 1)

                # Analyze peak times from timestamps
                peak_time_analysis = self._analyze_peak_times_from_logs(location_logs)

                # Create analytics entry
                entry = {
                    'property_id': source_id,
                    'property_name': location_name,
                    'source_type': source.get('source_type', 'Unknown'),
                    'current_people': int(avg_current_people),
                    'current_garbage': int(avg_current_garbage),
                    'avg_min_people': avg_min_people,
                    'avg_max_people': avg_max_people,
                    'avg_min_garbage': avg_min_garbage,
                    'avg_max_garbage': avg_max_garbage,
                    'total_logs': log_count,
                    'privacy': source.get('privacy', False),
                    'last_sync': latest_timestamp,
                    'status': source.get('status', 'stopped'),
                    'data_source': 'realtime_logs',
                    # Peak time analysis
                    'peak_times': peak_time_analysis.get('peak_times', []),
                    'busiest_hour': peak_time_analysis.get('busiest_hour', None),
                    'timestamps': timestamps,
                    # Range data for charts
                    'people_range': f"{int(avg_min_people)}-{int(avg_max_people)}",
                    'garbage_range': f"{int(avg_min_garbage)}-{int(avg_max_garbage)}"
                }

                analytics_data.append(entry)

        except Exception as e:
            st.error(f"Error collecting analytics data: {e}")
            import traceback
            traceback.print_exc()

        return analytics_data

    def _analyze_peak_times(self, log_entries):
        """Analyze multiple logs to identify peak times, patterns, and trends"""
        if not log_entries:
            return {
                'peak_times': [],
                'busiest_hour': 'N/A',
                'avg_daily_detections': 0,
                'weekly_pattern': {},
                'occupancy_trends': {}
            }

        # Initialize data structures for analysis
        hourly_detections = {}  # hour -> total detections
        daily_detections = {}   # date -> total detections
        weekly_pattern = {}     # day_of_week -> avg detections
        peak_times = []         # List of peak time periods

        for log in log_entries:
            try:
                timestamp_str = log.get('timestamp', '')
                if not timestamp_str:
                    continue

                # Parse timestamp
                if isinstance(timestamp_str, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        continue
                else:
                    timestamp = timestamp_str

                # Extract time components
                hour = timestamp.hour
                day_of_week = timestamp.strftime('%A')
                date = timestamp.date()

                # Aggregate detections by hour
                detections = log.get('person_detections', 0)
                if hour not in hourly_detections:
                    hourly_detections[hour] = []
                hourly_detections[hour].append(detections)

                # Aggregate by day
                if date not in daily_detections:
                    daily_detections[date] = 0
                daily_detections[date] += detections

                # Weekly pattern
                if day_of_week not in weekly_pattern:
                    weekly_pattern[day_of_week] = []
                weekly_pattern[day_of_week].append(detections)

            except Exception as e:
                continue  # Skip problematic log entries

        # Calculate averages and identify peaks
        busiest_hour = 'N/A'
        max_avg_detections = 0

        # Find busiest hour
        for hour, detections_list in hourly_detections.items():
            if detections_list:
                avg_detections = sum(detections_list) / len(detections_list)
                if avg_detections > max_avg_detections:
                    max_avg_detections = avg_detections
                    busiest_hour = f"{hour:02d}:00"

        # Calculate average daily detections
        avg_daily_detections = 0
        if daily_detections:
            total_daily = sum(daily_detections.values())
            avg_daily_detections = total_daily / len(daily_detections)

        # Process weekly patterns
        processed_weekly = {}
        for day, detections in weekly_pattern.items():
            if detections:
                processed_weekly[day] = sum(detections) / len(detections)

        # Identify peak time periods (hours with above-average activity)
        overall_avg = sum(sum(dets) / len(dets) for dets in hourly_detections.values() if dets) / len(hourly_detections) if hourly_detections else 0

        for hour in sorted(hourly_detections.keys()):
            detections_list = hourly_detections[hour]
            if detections_list:
                avg_for_hour = sum(detections_list) / len(detections_list)
                if avg_for_hour > overall_avg * 1.5:  # 50% above average
                    peak_times.append({
                        'hour': f"{hour:02d}:00",
                        'avg_detections': avg_for_hour,
                        'intensity': 'High' if avg_for_hour > overall_avg * 2 else 'Medium'
                    })

        # Sort peak times by average detections
        peak_times.sort(key=lambda x: x['avg_detections'], reverse=True)

        # Occupancy trends (simplified)
        occupancy_trends = {
            'total_logs_analyzed': len(log_entries),
            'avg_detection_rate': sum(log.get('person_detections', 0) / max(log.get('total_frames', 1), 1) for log in log_entries) / max(len(log_entries), 1),
            'peak_detection_rate': max((log.get('person_detections', 0) / max(log.get('total_frames', 1), 1) for log in log_entries), default=0),
            'consistency_score': len([log for log in log_entries if log.get('person_detections', 0) > 0]) / max(len(log_entries), 1)
        }

        return {
            'peak_times': peak_times[:5],  # Top 5 peak times
            'busiest_hour': busiest_hour,
            'avg_daily_detections': avg_daily_detections,
            'weekly_pattern': processed_weekly,
            'occupancy_trends': occupancy_trends
        }

    def _analyze_peak_times_from_logs(self, location_logs):
        """Analyze peak times from realtime database logs"""
        if not location_logs:
            return {
                'peak_times': [],
                'busiest_hour': None
            }

        # Initialize data structures
        hourly_counts = {}  # hour -> list of people counts
        busiest_hour = None
        max_avg_people = 0

        # Process each log entry
        for log_key, log_data in location_logs.items():
            timestamp_str = log_data.get('lastUpdate', '')
            people_count = log_data.get('currentNumberOfPeople', 0)

            if timestamp_str:
                try:
                    # Parse ISO timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    hour = timestamp.hour

                    if hour not in hourly_counts:
                        hourly_counts[hour] = []
                    hourly_counts[hour].append(people_count)
                except:
                    continue

        # Find busiest hour
        for hour, counts in hourly_counts.items():
            if counts:
                avg_people = sum(counts) / len(counts)
                if avg_people > max_avg_people:
                    max_avg_people = avg_people
                    busiest_hour = hour

        return {
            'peak_times': [busiest_hour] if busiest_hour is not None else [],
            'busiest_hour': busiest_hour
        }

    def _overview_dashboard(self, data):
        """Main overview dashboard with key metrics"""
        st.subheader("📊 Property Portfolio Overview")

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        total_properties = len(data)
        active_properties = len([d for d in data if d['status'] == 'running'])
        total_people = sum(d['current_people'] for d in data)
        total_garbage = sum(d['current_garbage'] for d in data)

        with col1:
            st.metric("Total Properties", total_properties)
        with col2:
            st.metric("Active Properties", f"{active_properties}/{total_properties}")
        with col3:
            st.metric("Total Current Occupancy", f"{total_people} people")
        with col4:
            st.metric("Total Waste Items", total_garbage)

        # Additional metrics from Firestore data
        col5, col6, col7, col8 = st.columns(4)

        total_frames_processed = sum(d.get('total_frames_processed', 0) for d in data)
        total_detections = sum(d.get('total_person_detections', 0) for d in data)
        avg_detection_rate = total_detections / total_frames_processed if total_frames_processed > 0 else 0
        properties_with_data = len([d for d in data if d.get('log_entries_count', 0) > 0])

        with col5:
            st.metric("Total Frames Processed", f"{total_frames_processed:,}")
        with col6:
            st.metric("Total Person Detections", f"{total_detections:,}")
        with col7:
            st.metric("Avg Detection Rate", f"{avg_detection_rate:.2f}")
        with col8:
            st.metric("Properties with Data", f"{properties_with_data}/{total_properties}")

        # Property status overview
        st.subheader("Property Status Overview")
        status_df = pd.DataFrame(data)

        if not status_df.empty:
            # Status distribution
            status_counts = status_df['status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Property Status Distribution",
                color_discrete_map={'running': 'green', 'stopped': 'red'}
            )
            st.plotly_chart(fig_status, use_container_width=True)

        # Current occupancy by property
        st.subheader("Current Occupancy by Property")
        if data:
            occupancy_data = [{
                'Property': d['property_name'],
                'Current Occupancy': d['current_people'],
                'Status': d['status']
            } for d in data]

            occupancy_df = pd.DataFrame(occupancy_data)
            fig_occupancy = px.bar(
                occupancy_df,
                x='Property',
                y='Current Occupancy',
                color='Status',
                title="Current Property Occupancy",
                color_discrete_map={'running': 'blue', 'stopped': 'gray'}
            )
            st.plotly_chart(fig_occupancy, use_container_width=True)

    def _property_performance(self, data):
        """Property performance analytics"""
        st.subheader("🏘️ Property Performance Analysis")

        if not data:
            st.info("No performance data available.")
            return

        # Performance metrics table
        perf_data = []
        for d in data:
            # Calculate performance score based on Firestore data
            avg_people = d.get('avg_people_per_frame', 0)
            max_people = d.get('max_people_in_frame', 0)
            detection_rate = d.get('total_person_detections', 0) / max(d.get('total_frames_processed', 1), 1)
            log_count = d.get('log_entries_count', 0)

            # Performance score considers detection activity and data completeness
            activity_score = min(100, detection_rate * 10)  # Scale detection rate
            data_completeness = min(100, log_count * 10)  # More logs = more complete data
            performance_score = (activity_score + data_completeness) / 2

            perf_data.append({
                'Property': d['property_name'],
                'Status': d['status'],
                'Avg People/Frame': avg_people,
                'Max People/Frame': max_people,
                'Detection Rate': detection_rate,
                'Data Points': log_count,
                'Performance Score': performance_score,
                'Last Updated': d['last_sync'].strftime('%H:%M:%S') if d['last_sync'] else 'Never'
            })

        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)

        # Performance radar chart
        if len(data) > 1:
            st.subheader("Property Performance Comparison")
            categories = ['Detection Activity', 'Data Completeness', 'Peak Capacity', 'Current Occupancy']

            fig = go.Figure()

            for d in data:
                detection_activity = min(100, (d.get('total_person_detections', 0) / d.get('total_frames_processed', 1)) * 10)
                data_completeness = min(100, d.get('log_entries_count', 0) * 10)
                peak_capacity = min(100, d.get('max_people_in_frame', 0) * 10)
                current_occupancy = min(100, d.get('current_people', 0) * 5)

                fig.add_trace(go.Scatterpolar(
                    r=[detection_activity, data_completeness, peak_capacity, current_occupancy],
                    theta=categories,
                    fill='toself',
                    name=d['property_name']
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Property Performance Comparison (Firestore Data)"
            )
            st.plotly_chart(fig, use_container_width=True)

    def _waste_management(self, data):
        """Waste management analytics"""
        st.subheader("🗑️ Waste Management Insights")

        if not data:
            st.info("No waste data available.")
            return

        # Waste distribution
        waste_data = [{
            'Property': d['property_name'],
            'Waste Items': d['current_garbage'],
            'Status': d['status']
        } for d in data if d['current_garbage'] > 0]

        if waste_data:
            waste_df = pd.DataFrame(waste_data)

            col1, col2 = st.columns(2)

            with col1:
                fig_waste = px.bar(
                    waste_df,
                    x='Property',
                    y='Waste Items',
                    color='Status',
                    title="Waste Distribution by Property"
                )
                st.plotly_chart(fig_waste, use_container_width=True)

            with col2:
                # Waste efficiency score
                waste_efficiency = []
                for d in data:
                    efficiency = max(0, 100 - (d['current_garbage'] * 10))
                    waste_efficiency.append({
                        'Property': d['property_name'],
                        'Efficiency': efficiency,
                        'Waste Items': d['current_garbage']
                    })

                efficiency_df = pd.DataFrame(waste_efficiency)
                fig_efficiency = px.bar(
                    efficiency_df,
                    x='Property',
                    y='Efficiency',
                    title="Waste Management Efficiency",
                    color='Waste Items',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig_efficiency, use_container_width=True)

    def _occupancy_trends(self, data):
        """Occupancy trends and patterns"""
        st.subheader("👥 Occupancy Trends & Patterns")

        if not data:
            st.info("No occupancy data available.")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Peak occupancy analysis based on Firestore data
            st.info("🏆 Peak Performance Properties")
            sorted_by_max_people = sorted(data, key=lambda x: x.get('max_people_in_frame', 0), reverse=True)
            peak_props = sorted_by_max_people[:3]
            for prop in peak_props[:3]:
                max_people = prop.get('max_people_in_frame', 0)
                avg_people = prop.get('avg_people_per_frame', 0)
                st.write(f"• {prop['property_name']}: Max {max_people}, Avg {avg_people:.1f} people/frame")

        with col2:
            # Detection efficiency
            st.info("🎯 Detection Efficiency")
            total_frames = sum(d.get('total_frames_processed', 0) for d in data)
            total_detections = sum(d.get('total_person_detections', 0) for d in data)
            avg_detection_rate = total_detections / total_frames if total_frames > 0 else 0
            st.metric("Average Detection Rate", f"{avg_detection_rate:.2f}")
            st.metric("Average Data Points", f"{len(data):.1f}")

        # Recommendations
        st.subheader("💡 Management Recommendations")

        low_occupancy = [d for d in data if d['current_people'] < 5 and d['status'] == 'running']
        if low_occupancy:
            st.warning("🏢 Low occupancy properties:")
            for prop in low_occupancy:
                st.write(f"• **{prop['property_name']}**: Only {prop['current_people']} people - Consider marketing or events")

        high_occupancy = [d for d in data if d['current_people'] > 30]
        if high_occupancy:
            st.success("🔥 High occupancy properties:")
            for prop in high_occupancy:
                st.write(f"• **{prop['property_name']}**: {prop['current_people']} people - Excellent performance!")

    def _peak_times_analysis(self, data):
        """Analyze peak times, patterns, and temporal trends"""
        st.subheader("⏰ Peak Times & Activity Patterns")

        if not data:
            st.info("No data available for peak time analysis.")
            return

        # Filter properties with sufficient data
        properties_with_data = [d for d in data if d.get('log_entries_count', 0) > 0]

        if not properties_with_data:
            st.warning("No properties have sufficient historical data for peak time analysis.")
            return

        # Overall peak time analysis
        st.markdown("### 📈 Overall Peak Time Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Busiest hours across all properties
            st.info("🏆 Busiest Hours")
            busiest_hours = {}
            for d in properties_with_data:
                busiest = d.get('busiest_hour', 'N/A')
                if busiest != 'N/A':
                    if busiest not in busiest_hours:
                        busiest_hours[busiest] = []
                    busiest_hours[busiest].append(d['property_name'])

            if busiest_hours:
                for hour, properties in sorted(busiest_hours.items()):
                    st.write(f"• **{hour}**: {', '.join(properties)}")
            else:
                st.write("No peak hour data available")

        with col2:
            # Average daily patterns
            st.info("📊 Daily Activity Summary")
            total_avg_daily = sum(d.get('avg_daily_detections', 0) for d in properties_with_data)
            avg_daily_overall = total_avg_daily / len(properties_with_data) if properties_with_data else 0
            st.metric("Average Daily Detections", f"{avg_daily_overall:.1f}")

            # Most active property
            most_active = max(properties_with_data, key=lambda x: x.get('avg_daily_detections', 0))
            st.metric("Most Active Property", most_active['property_name'])

        # Weekly patterns
        st.markdown("### 📅 Weekly Activity Patterns")

        # Collect all weekly patterns
        all_weekly_patterns = {}
        for d in properties_with_data:
            weekly = d.get('weekly_pattern', {})
            for day, avg_detections in weekly.items():
                if day not in all_weekly_patterns:
                    all_weekly_patterns[day] = []
                all_weekly_patterns[day].append(avg_detections)

        # Calculate average for each day
        avg_weekly = {}
        for day, detections in all_weekly_patterns.items():
            if detections:
                avg_weekly[day] = sum(detections) / len(detections)

        if avg_weekly:
            # Sort days in proper order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sorted_weekly = {day: avg_weekly.get(day, 0) for day in day_order if day in avg_weekly}

            fig_weekly = px.bar(
                x=list(sorted_weekly.keys()),
                y=list(sorted_weekly.values()),
                title="Average Weekly Activity Pattern",
                labels={'x': 'Day of Week', 'y': 'Average Detections'}
            )
            st.plotly_chart(fig_weekly, use_container_width=True)

        # Peak time periods analysis
        st.markdown("### ⏰ Peak Time Periods")

        all_peak_times = []
        for d in properties_with_data:
            peak_times = d.get('peak_times', [])
            for peak in peak_times:
                all_peak_times.append({
                    'property': d['property_name'],
                    'hour': peak.get('hour', 'N/A'),
                    'avg_detections': peak.get('avg_detections', 0),
                    'intensity': peak.get('intensity', 'Low')
                })

        if all_peak_times:
            # Sort by average detections
            all_peak_times.sort(key=lambda x: x['avg_detections'], reverse=True)

            # Display top peak times
            peak_df = pd.DataFrame(all_peak_times[:10])  # Top 10 peak times
            st.dataframe(peak_df, use_container_width=True)

            # Peak time distribution by hour
            peak_hours = {}
            for peak in all_peak_times:
                hour = peak['hour']
                if hour not in peak_hours:
                    peak_hours[hour] = 0
                peak_hours[hour] += 1

            if peak_hours:
                fig_peaks = px.bar(
                    x=list(peak_hours.keys()),
                    y=list(peak_hours.values()),
                    title="Peak Time Distribution by Hour",
                    labels={'x': 'Hour', 'y': 'Number of Peak Periods'}
                )
                st.plotly_chart(fig_peaks, use_container_width=True)

        # Occupancy trends analysis
        st.markdown("### 📈 Occupancy Trends & Consistency")

        trends_data = []
        for d in properties_with_data:
            trends = d.get('occupancy_trends', {})
            trends_data.append({
                'Property': d['property_name'],
                'Logs Analyzed': trends.get('total_logs_analyzed', 0),
                'Avg Detection Rate': f"{trends.get('avg_detection_rate', 0):.3f}",
                'Peak Detection Rate': f"{trends.get('peak_detection_rate', 0):.3f}",
                'Consistency Score': f"{trends.get('consistency_score', 0):.1f}"
            })

        if trends_data:
            trends_df = pd.DataFrame(trends_data)
            st.dataframe(trends_df, use_container_width=True)

            # Consistency analysis
            avg_consistency = sum(trends.get('consistency_score', 0) for d in properties_with_data) / len(properties_with_data) if properties_with_data else 0
            st.metric("Average Data Consistency", f"{avg_consistency:.1f}")

            # Recommendations based on consistency
            low_consistency = [row for row in trends_data if float(row['Consistency Score']) < 0.7]
            if low_consistency:
                st.warning("⚠️ **Inconsistent Data Properties**: " + ", ".join([row['Property'] for row in low_consistency]))
                st.info("Consider reviewing detection processes for these properties.")

    def _management_insights(self, data):
        """Actionable management insights and recommendations"""
        st.subheader("💡 Property Management Insights & Recommendations")

        if not data:
            st.info("No data available for insights.")
            return

        # Cleaning Schedule Recommendations based on Firestore data
        st.markdown("### 🧹 Cleaning Schedule Recommendations")

        cleaning_recommendations = []
        for d in data:
            waste_level = d.get('current_garbage', 0)
            avg_occupancy = d.get('avg_people_per_frame', 0)
            max_occupancy = d.get('max_people_in_frame', 0)
            detection_rate = d.get('total_person_detections', 0) / max(d.get('total_frames_processed', 1), 1)

            # Enhanced cleaning logic based on Firestore data
            # High detection rate + high occupancy = more frequent cleaning needed
            activity_factor = detection_rate * avg_occupancy

            if activity_factor > 50 or waste_level >= 3:
                frequency = "Daily cleaning required"
                priority = "🔴 High Priority"
                staff_needed = max(2, min(4, int(activity_factor / 10)))
            elif activity_factor > 25 or waste_level >= 1:
                frequency = "Twice daily cleaning"
                priority = "🟡 Medium Priority"
                staff_needed = max(1, int(activity_factor / 15))
            else:
                frequency = "Regular cleaning schedule"
                priority = "🟢 Low Priority"
                staff_needed = 1

            # Adjust for peak occupancy
            if max_occupancy > 15:
                staff_needed = max(staff_needed, 2)
                frequency = "Intensive " + frequency.lower()

            cleaning_recommendations.append({
                'Property': d['property_name'],
                'Activity Factor': f"{activity_factor:.1f}",
                'Waste Level': waste_level,
                'Peak Occupancy': max_occupancy,
                'Cleaning Frequency': frequency,
                'Priority': priority,
                'Staff Needed': staff_needed,
                'Estimated Cost': f"${staff_needed * 15}/day"
            })

        cleaning_df = pd.DataFrame(cleaning_recommendations)
        st.dataframe(cleaning_df, use_container_width=True)

        # Staffing Recommendations
        st.markdown("### 👥 Staffing Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Security Staffing**")
            security_needs = []
            for d in data:
                # Base security needs on Firestore data
                avg_occupancy = d.get('avg_people_per_frame', 0)
                max_occupancy = d.get('max_people_in_frame', 0)
                detection_rate = d.get('total_person_detections', 0) / max(d.get('total_frames_processed', 1), 1)

                # Calculate security needs based on peak occupancy and activity
                base_staff = 1
                if max_occupancy > 10 or avg_occupancy > 5:
                    base_staff = 2
                if max_occupancy > 20 or detection_rate > 5:
                    base_staff = 3

                # Add extra for high-activity areas
                if detection_rate > 8 or 'market' in d['property_name'].lower():
                    base_staff += 1

                security_needs.append({
                    'Property': d['property_name'],
                    'Avg Occupancy': f"{avg_occupancy:.1f}",
                    'Peak Occupancy': max_occupancy,
                    'Activity Rate': f"{detection_rate:.2f}",
                    'Recommended Security': base_staff
                })

            security_df = pd.DataFrame(security_needs)
            st.dataframe(security_df, use_container_width=True)

        with col2:
            st.markdown("**Maintenance Staffing**")
            maintenance_needs = []
            for d in data:
                # Maintenance based on Firestore data and activity
                base_maintenance = 1
                waste_level = d.get('current_garbage', 0)
                detection_rate = d.get('total_person_detections', 0) / max(d.get('total_frames_processed', 1), 1)
                max_occupancy = d.get('max_people_in_frame', 0)

                # Higher maintenance for high-activity, high-waste areas
                if waste_level > 2 or detection_rate > 6:
                    base_maintenance += 1

                # Larger properties need more maintenance
                if max_occupancy > 15:
                    base_maintenance += 1

                maintenance_needs.append({
                    'Property': d['property_name'],
                    'Activity Rate': f"{detection_rate:.2f}",
                    'Waste Issues': waste_level,
                    'Peak Occupancy': max_occupancy,
                    'Maintenance Staff': base_maintenance
                })

            maintenance_df = pd.DataFrame(maintenance_needs)
            st.dataframe(maintenance_df, use_container_width=True)

        # Business Insights based on Firestore data
        st.markdown("### 📈 Business Intelligence Insights")

        # Revenue potential analysis using Firestore metrics
        total_avg_occupancy = sum(d.get('avg_people_per_frame', 0) for d in data)
        total_max_occupancy = sum(d.get('max_people_in_frame', 0) for d in data)
        total_frames = sum(d.get('total_frames_processed', 0) for d in data)
        total_detections = sum(d.get('total_person_detections', 0) for d in data)

        col1, col2, col3 = st.columns(3)

        with col1:
            # Foot traffic analysis
            st.metric("Total Avg Occupancy", f"{total_avg_occupancy:.1f}")
            st.metric("Peak Capacity Sum", total_max_occupancy)

        with col2:
            # Detection efficiency
            detection_efficiency = total_detections / total_frames if total_frames > 0 else 0
            st.metric("Overall Detection Rate", f"{detection_efficiency:.2f}")
            data_completeness = len([d for d in data if d.get('log_entries_count', 0) > 0]) / len(data) * 100
            st.metric("Data Completeness", f"{data_completeness:.1f}")

        with col3:
            # Activity analysis
            avg_activity = total_detections / len(data) if data else 0
            st.metric("Avg Detections/Property", f"{avg_activity:.1f}")
            high_activity_props = len([d for d in data if d.get('total_person_detections', 0) > 50])
            st.metric("High Activity Properties", f"{high_activity_props}/{len(data)}")

        # Actionable recommendations based on Firestore data
        st.markdown("### 🎯 Action Items")

        recommendations = []

        # High activity areas needing attention
        high_activity = [d for d in data if d.get('total_person_detections', 0) > 100]
        if high_activity:
            recommendations.append(f"🚨 **High Activity Alert**: {len(high_activity)} properties showing intense activity - consider additional staffing")

        # Data quality issues
        low_data = [d for d in data if d.get('log_entries_count', 0) < 5 and d['status'] == 'running']
        if low_data:
            recommendations.append(f"� **Data Quality Check**: {len(low_data)} active properties have limited data - verify detection processes")

        # Peak capacity concerns
        high_peak = [d for d in data if d.get('max_people_in_frame', 0) > 25]
        if high_peak:
            recommendations.append(f"🏢 **Capacity Planning**: {len(high_peak)} properties frequently reach high capacity - consider expansion")

        # Cleaning priorities based on activity
        cleaning_priority = [d for d in data if (d.get('total_person_detections', 0) / max(d.get('total_frames_processed', 1), 1)) > 7]
        if cleaning_priority:
            recommendations.append(f"🧹 **Cleaning Priority**: {len(cleaning_priority)} high-activity properties need intensive cleaning schedules")

        # Staffing alerts based on Firestore data
        total_security = sum(max(1, min(3, int(d.get('max_people_in_frame', 0) / 8) + 1)) for d in data)
        total_cleaning = sum(max(1, int((d.get('total_person_detections', 0) / max(d.get('total_frames_processed', 1), 1)) / 3)) for d in data)

        recommendations.append(f"👥 **Staffing Requirements**: {total_security} security personnel, {total_cleaning} cleaning staff recommended across portfolio")

        for rec in recommendations:
            st.info(rec)

        # Cost analysis based on Firestore-derived staffing needs
        st.markdown("### 💰 Cost Analysis")

        # Calculate costs based on activity-driven staffing
        security_cost = sum(max(1, min(3, int(d.get('max_people_in_frame', 0) / 8) + 1)) * 20 for d in data)
        cleaning_cost = sum(max(1, int((d.get('total_person_detections', 0) / max(d.get('total_frames_processed', 1), 1)) / 3)) * 15 for d in data)
        maintenance_cost = sum(max(1, int(d.get('max_people_in_frame', 0) / 15) + 1) * 25 for d in data)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Cleaning Cost (Daily)", f"${cleaning_cost}")
        with col2:
            st.metric("Security Cost (Daily)", f"${security_cost}")
        with col3:
            st.metric("Maintenance Cost (Daily)", f"${maintenance_cost}")

        total_cost = cleaning_cost + security_cost + maintenance_cost
        st.metric("Total Daily Cost", f"${total_cost}")

        # Cost efficiency analysis
        st.markdown("#### Cost Efficiency Insights")
        if data:
            avg_cost_per_property = total_cost / len(data)
            avg_occupancy = sum(d.get('avg_people_per_frame', 0) for d in data) / len(data)
            cost_per_person = total_cost / (avg_occupancy * len(data)) if avg_occupancy > 0 else 0

            st.metric("Average Cost per Property", f"${avg_cost_per_property:.0f}")
            st.metric("Cost per Person per Day", f"${cost_per_person:.2f}")
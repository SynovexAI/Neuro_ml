export default function EtlLabDesktopUI() {
  return (
    <section className="etl-dashboard-screen">
      <div className="etl-dashboard-frame">
        <aside className="etl-sidebar">
          <div className="etl-brand">
            <span className="brand-mark">N</span>
            <span className="brand-name">TalentAI</span>
          </div>

          <button className="candidate-button">Candidate</button>

          <nav className="etl-nav">
            <div className="nav-section-title">MAIN</div>
            <a className="nav-link active" href="#">
              <span className="nav-icon">▦</span>
              <span>Overview</span>
            </a>
            <a className="nav-link" href="#">
              <span className="nav-icon">◎</span>
              <span>Profile</span>
            </a>
            <a className="nav-link" href="#">
              <span className="nav-icon">◌</span>
              <span>Applications</span>
            </a>
            <a className="nav-link" href="#">
              <span className="nav-icon">✧</span>
              <span>Verification Center</span>
            </a>
          </nav>

          <nav className="etl-nav lower">
            <div className="nav-section-title">AI RESUME</div>
            <a className="nav-link" href="#">
              <span className="nav-icon">✳</span>
              <span>Resume Builder</span>
            </a>
            <a className="nav-link" href="#">
              <span className="nav-icon">▤</span>
              <span>Upload Resume</span>
            </a>
            <a className="nav-link" href="#">
              <span className="nav-icon">◈</span>
              <span>Resume Simulation</span>
            </a>
            <a className="nav-link" href="#">
              <span className="nav-icon">◍</span>
              <span>Resume Screening</span>
            </a>
            <a className="nav-link" href="#">
              <span className="nav-icon">◇</span>
              <span>Resume Improvement</span>
            </a>
            <a className="nav-link" href="#">
              <span className="nav-icon">✹</span>
              <span>Dynamic Resume</span>
            </a>
          </nav>

          <nav className="etl-nav lower">
            <div className="nav-section-title">AI TOOLS</div>
            <a className="nav-link" href="#">
              <span className="nav-icon">✎</span>
              <span>AI Project Workspace</span>
            </a>
            <a className="nav-link" href="#">
              <span className="nav-icon">◉</span>
              <span>Interview Prep</span>
            </a>
          </nav>
        </aside>

        <main className="etl-content">
          <section className="etl-kpi-row">
            <article className="kpi-card kpi-card-1">
              <div className="kpi-heading">
                <span className="kpi-icon">⌕</span>
                <span className="kpi-title">Search...</span>
              </div>
              <div className="kpi-value">87</div>
              <div className="kpi-foot">+5% this week</div>
            </article>

            <article className="kpi-card">
              <div className="kpi-heading top-only">
                <span className="kpi-title">Applications</span>
              </div>
              <div className="kpi-value">0</div>
              <div className="kpi-foot">2 pending</div>
            </article>

            <article className="kpi-card">
              <div className="kpi-heading top-only">
                <span className="kpi-title">Learning Score</span>
              </div>
              <div className="kpi-value">78</div>
              <div className="kpi-foot">+3% this month</div>
            </article>

            <article className="kpi-card kpi-score">
              <div className="score-row">
                <span className="score-badge">SK</span>
                <span className="score-badge score-badge-blue">N</span>
                <span className="score-name">Naresh Cumar K</span>
              </div>
              <div className="score-value">92%</div>
              <div className="score-foot">Top 15%</div>
            </article>
          </section>

          <section className="etl-profile-area">
            <article className="profile-card glass-panel">
              <div className="profile-top">
                <span className="linkedin-status">
                  <span className="mini-icon linkedin-icon">in</span>
                  <span>LinkedIn</span>
                  <span className="small-badge disconnected">Not Connected</span>
                </span>
                <span className="small-badge connected">OAuth 2.0 Connected</span>
              </div>

              <div className="profile-main">
                <div className="avatar-block">
                  <span className="avatar-avatar">☻</span>
                  <div>
                    <h2 className="guest-title">Guest Candidate</h2>
                    <div className="guest-mail">nareshcumar18@gmail.com</div>
                    <div className="account-status">Account Status: Inactive</div>
                  </div>
                </div>

                <div className="last-synced">Last Synced: Never</div>
                <button className="connect-button">Connect LinkedIn</button>
              </div>
            </article>

            <article className="summary-card glass-panel">
              <div className="summary-header">
                <span className="title-icon">AI</span>
                <span className="summary-title">AI Executive Profile Summary</span>
              </div>
              <p className="summary-copy">
                Please connect your LinkedIn profile to synthesize a tailored AI candidate summary for recruiter evaluation.
              </p>
              <p className="summary-copy muted-copy">
                Generated automatically over every profile sync checkpoint.
              </p>
            </article>
          </section>

          <section className="recommendations-panel glass-panel">
            <div className="recommendation-head">
              <h2>Top 5 Recommended Jobs</h2>
              <button className="view-jobs-button">View All Jobs</button>
            </div>
            <div className="empty-state">
              <span className="empty-icon">♣</span>
              <h3>No recommendations yet</h3>
              <p>Complete your profile to get personalized job matches</p>
            </div>
          </section>
        </main>
      </div>
    </section>
  );
}
